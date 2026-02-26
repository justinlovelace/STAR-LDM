"""
Generate text with STAR-LDM.

Batch mode:
    python scripts/generate.py --model_path PATH \\
        --prompts "The movie was" "Once upon a time"

Interactive mode:
    python scripts/generate.py --model_path PATH --interactive

Classifier-guided generation:
    python scripts/generate.py --model_path PATH \\
        --classifier_path PATH --cls_guidance 3.0 --cls_target 1.0 \\
        --prompts "The movie was"
"""

import argparse
import sys

from star_ldm.interface import TransfusionGPTInterface


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate text with STAR-LDM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to STAR-LDM checkpoint directory or .pt file')
    parser.add_argument('--device', type=str, default='cuda')

    # Mode
    parser.add_argument('--prompts', type=str, nargs='+',
                        help='One or more prompts for batch generation')
    parser.add_argument('--interactive', action='store_true',
                        help='Enter interactive REPL mode')

    # Classifier guidance
    parser.add_argument('--classifier_path', type=str, default=None,
                        help='Path to a pretrained classifier checkpoint for guided generation')
    parser.add_argument('--cls_guidance', type=float, default=0.0,
                        help='Classifier guidance scale (0 = disabled)')
    parser.add_argument('--cls_target', type=float, default=None,
                        help='Target class for guidance (0.0 or 1.0)')

    # Sampling
    parser.add_argument('--sampling_timesteps', type=int, default=50,
                        help='Number of diffusion sampling steps')
    parser.add_argument('--sampler', type=str, default='ddpm', choices=['ddpm', 'ddim'])
    parser.add_argument('--cls_free_guidance', type=float, default=1.0,
                        help='Classifier-free guidance scale')

    # Token generation
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--repetition_penalty', type=float, default=1.2)

    args = parser.parse_args()
    if not args.prompts and not args.interactive:
        parser.error('Provide --prompts or --interactive')
    return args


def main():
    args = parse_args()

    print('Loading model...')
    interface = TransfusionGPTInterface(
        model_path=args.model_path,
        device=args.device,
        classifier_path=args.classifier_path,
    )

    sample_kwargs = dict(
        sampling_timesteps=args.sampling_timesteps,
        sampler=args.sampler,
        cls_free_guidance=args.cls_free_guidance,
        cls_guidance=args.cls_guidance,
        cls_target=args.cls_target,
        generate_kwargs=dict(
            do_sample=True,
            num_beams=1,
            pad_token_id=interface.tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        ),
    )

    if args.interactive:
        print('\nSTAR-LDM Interactive Demo')
        if args.classifier_path and args.cls_guidance != 0:
            print(f'  Classifier guidance: scale={args.cls_guidance}, target={args.cls_target}')
        print(f'  Diffusion steps: {args.sampling_timesteps} ({args.sampler})')
        print('  Type "quit" to exit.\n')

        while True:
            try:
                prompt = input('Prompt> ')
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.strip().lower() == 'quit':
                break
            if not prompt.strip():
                continue

            generation = interface.generate([prompt], **sample_kwargs)[0]
            print(f'\n{generation}\n')
    else:
        generations = interface.generate(args.prompts, **sample_kwargs)
        for prompt, generation in zip(args.prompts, generations):
            print(f'\nPrompt: {prompt}')
            print(f'Generation: {generation}')


if __name__ == '__main__':
    main()
