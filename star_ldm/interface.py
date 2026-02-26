import torch
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
from omegaconf import OmegaConf

from star_ldm.models.transfusion import TransfusionGPT, variance_preserving_map
from star_ldm.diffusion.noise_schedule import log_snr_to_alpha2

class TransfusionGPTInterface:
    def __init__(self, model_path: str, device: str = 'cuda', classifier_path: Optional[str] = None):
        """
        Args:
            model_path: Path to the STAR-LDM checkpoint directory or ``.pt`` file.
            device: Device to load models onto.
            classifier_path: Optional path to a pretrained
                :class:`~star_ldm.models.classifier.NoiseConditionedMLP` checkpoint
                for classifier-guided generation.
        """
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.gpt2.config._name_or_path)

        self.classifier = None
        if classifier_path is not None:
            from star_ldm.models.classifier import load_classifier
            self.classifier = load_classifier(classifier_path, device=str(self.device))

    def _load_model(self, model_path: str) -> 'TransfusionGPT':
        # Check if model_path ends in '.pt'
        if model_path.endswith('.pt'):
            model_dir = os.path.dirname(model_path)
        else:
            model_dir = model_path
            model_path = os.path.join(model_dir, 'model.pt')

        # Grab model directory from model_path
        transfusion_cfg = OmegaConf.load(os.path.join(model_dir, 'args.yaml'))

        model = TransfusionGPT(
            dataset_name=transfusion_cfg.dataset_name,
            transfusion_cfg=transfusion_cfg,
            gpt2_model_name=transfusion_cfg.train.lm_name,
            gamma_min=-15,
            gamma_max=15,
            clf_guidance_dropout=0.1,
            scale_by_std=True,
            global_norm=transfusion_cfg.train.get('global_norm', False),
        )

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        if isinstance(ckpt, dict) and 'ema' in ckpt:
            # Direct training checkpoint — extract EMA weights
            from ema_pytorch import EMA
            ema = EMA(model, beta=0.999, update_every=10, power=3/4, update_after_step=1000)
            ema.load_state_dict(ckpt['ema'], strict=False)
            model = ema.ema_model.to(self.device)
        else:
            # Plain state_dict
            state_dict = ckpt
            model.load_state_dict(state_dict, strict=False)
            model = model.to(self.device)

        return model

    def generate(self, prompts: List[str], cls_guidance: float = 0.0,
                 cls_target: Optional[float] = None, **kwargs) -> List[str]:
        """
        Generate text for a list of prompts.

        Args:
            prompts: List of prompts to generate from.
            cls_guidance: Classifier guidance scale. ``0.0`` disables guidance.
                Positive values steer toward ``cls_target``.
            cls_target: Target class for classifier guidance (``0.0`` or ``1.0``).
                Required when ``cls_guidance != 0``.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`TransfusionGPT.sample`.

        Returns:
            List of generated text strings.
        """
        if cls_guidance != 0.0:
            if self.classifier is None:
                raise ValueError(
                    "Classifier guidance requested but no classifier loaded. "
                    "Pass classifier_path when constructing TransfusionGPTInterface."
                )
            if cls_target is None:
                raise ValueError(
                    "cls_target must be specified (0.0 or 1.0) when using classifier guidance."
                )

        generations = []
        generate_kwargs = kwargs.pop('generate_kwargs', {})
        for prompt in tqdm(prompts, desc="Generating"):
            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
            _, generation = self.model.sample(
                input_ids,
                cls_guidance=cls_guidance,
                classifier=self.classifier,
                cls_target=cls_target,
                generate_kwargs=generate_kwargs,
                **kwargs,
            )
            generations.extend(generation)
        return generations

    def interactive_demo(self, generate_kwargs: Optional[Dict[str, Any]] = None):
        """
        Run an interactive demo allowing the user to try different generation settings.
        """
        print("STAR-LDM Interactive Demo")
        print("Enter 'quit' to exit")

        while True:
            prompt = input("\nEnter a prompt: ")
            if prompt.lower() == 'quit':
                break

            if generate_kwargs is None:
                generation = self.generate([prompt])[0]
            else:
                generation = self.generate([prompt], **generate_kwargs)[0]

            print(f"Generated text: {generation}")
