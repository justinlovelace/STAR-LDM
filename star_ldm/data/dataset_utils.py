import torch
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datasets import load_dataset

from star_ldm.data.CONSTANTS import MAX_LENGTH


@dataclass
class DataCollatorWithDiffusionTokens:
    """
    Data collator for language modeling with diffusion prompts.

    For each example, randomly splits the token sequence into a prompt and continuation,
    inserts ``num_diffusion_tokens`` pad tokens between them, and builds the corresponding
    labels (with prompt + diffusion positions masked to -100).

    Input: list of dicts with ``input_ids`` (and optionally ``num_tokens`` / ``num_prompt_tokens``).
    Output dict keys: ``input_ids``, ``labels``, ``attention_mask``, ``diffusion_token_mask``,
    ``continuation_start``, ``prompt`` (str), ``continuation`` (str).
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = MAX_LENGTH
    return_tensors: str = "pt"
    num_diffusion_tokens: int = 8
    validation: bool = False
    min_continuation_len: int = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if 'num_prompt_tokens' in features[0]:
            continuation_start = torch.tensor(
                [feature['num_prompt_tokens'] for feature in features], dtype=torch.long)
        elif 'num_tokens' in features[0]:
            continuation_start = torch.rand(size=(len(features),))
            num_tokens = torch.stack([feature['num_tokens'] for feature in features])
            continuation_start = torch.round(continuation_start * (num_tokens - 1)).long()
        else:
            continuation_start = torch.rand(size=(len(features),))
            num_tokens = torch.tensor(
                [feature['input_ids'][:self.max_length].shape[0] for feature in features],
                dtype=torch.long, device=features[0]['input_ids'].device)
            continuation_start = torch.round(
                continuation_start * (num_tokens - self.min_continuation_len)).long()

        prompts = [features[idx]['input_ids'][:self.max_length][:continuation_start[idx].item()]
                   for idx in range(len(features))]
        continuations = [features[idx]['input_ids'][:self.max_length][continuation_start[idx].item():]
                         for idx in range(len(features))]

        diffusion_token_prompt = torch.tensor(
            [self.tokenizer.pad_token_id] * self.num_diffusion_tokens,
            dtype=prompts[0].dtype, device=prompts[0].device)
        input_ids = [torch.cat((prompts[idx], diffusion_token_prompt, continuations[idx]), dim=0)
                     for idx in range(len(features))]

        batch = self.tokenizer.pad(
            {'input_ids': input_ids},
            padding='longest',
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            return_attention_mask=True,
        )

        batch_size, seq_len = batch['input_ids'].shape

        seq_ids = torch.arange(seq_len).repeat((batch_size, 1))
        batch['diffusion_token_mask'] = torch.logical_and(
            (seq_ids >= continuation_start[:, None]),
            (seq_ids < continuation_start[:, None] + self.num_diffusion_tokens))

        batch["labels"] = batch["input_ids"].clone()
        batch['labels'][batch['labels'] == self.tokenizer.pad_token_id] = -100


        batch['continuation_start'] = continuation_start

        batch['prompt'] = [self.tokenizer.decode(prompt, skip_special_tokens=True) for prompt in prompts]
        batch['continuation'] = [self.tokenizer.decode(continuation, skip_special_tokens=True)
                                 for continuation in continuations]

        if self.validation:
            clean_batch = self.tokenizer.pad(
                {'input_ids': [feature['input_ids'] for feature in features]},
                padding='longest',
                max_length=self.max_length,
                return_tensors=self.return_tensors,
                return_attention_mask=True,
            )
            batch['clean_input_ids'] = clean_batch['input_ids']
            batch['clean_labels'] = clean_batch['input_ids'].clone()
            batch['clean_labels'][batch['clean_labels'] == self.tokenizer.pad_token_id] = -100

        return batch


def get_fineweb_streaming_dataset(
    tokenizer_name: str = 'gpt2-large',
    subset: str = 'sample-10BT',
    max_length: int = MAX_LENGTH,
    min_chunk_length: int = 64,
    buffer_size: int = 10_000,
):
    """
    Stream FineWeb from HuggingFace Hub, tokenize on-the-fly, and chunk into
    fixed-length sequences.

    Args:
        tokenizer_name: HuggingFace tokenizer name (e.g. ``'gpt2-large'``).
        subset: FineWeb subset name (``'sample-10BT'`` or ``'sample-100BT'``).
        max_length: Context length for chunking (tokens per example).
        min_chunk_length: Minimum tokens to keep a partial chunk.
        buffer_size: Shuffle buffer size for the streaming dataset.

    Returns:
        A HuggingFace ``IterableDataset`` yielding dicts with ``input_ids`` tensors.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    dataset = load_dataset(
        'HuggingFaceFW/fineweb',
        name=subset,
        split='train',
        streaming=True,
    )

    def tokenize_and_chunk(examples):
        tokenized = tokenizer(
            examples['text'],
            truncation=False,
            padding=False,
        )

        all_input_ids = []
        for ids in tokenized['input_ids']:
            for i in range(0, len(ids) - max_length + 1, max_length):
                all_input_ids.append(ids[i:i + max_length])
            remaining = len(ids) % max_length
            if remaining >= min_chunk_length:
                all_input_ids.append(ids[-remaining:])

        return {'input_ids': all_input_ids}

    dataset = dataset.map(
        tokenize_and_chunk,
        batched=True,
        remove_columns=dataset.column_names,
    )
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.with_format('pt')
    return dataset


def get_c4_validation_dataset(
    tokenizer_name: str = 'gpt2-large',
    max_length: int = MAX_LENGTH,
    num_examples: int = 5000,
):
    """
    Load a small C4 validation split for periodic evaluation during training.

    Returns:
        A HuggingFace ``IterableDataset`` yielding dicts with ``input_ids`` tensors.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    dataset = load_dataset(
        'allenai/c4', 'en', split='validation', streaming=True,
    ).shuffle(seed=42, buffer_size=50_000).take(num_examples)

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples['text'], max_length=max_length, truncation=True)
        num_tokens = [len(ids) for ids in tokenized['input_ids']]
        tokenized['num_tokens'] = num_tokens
        return tokenized

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.with_format('pt')
    return dataset
