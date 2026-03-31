#!/usr/bin/env python3
"""Command-line interface for the collective encoder package."""

import argparse
import sys
from pathlib import Path

def main_train():
    """Entry point for the collective-encoder-train command."""
    from collective_encoder.engine import main as engine_main

    # Temporarily modify sys.argv to pass arguments to engine
    original_argv = sys.argv.copy()
    try:
        # Parse our CLI args first
        parser = argparse.ArgumentParser(
            description="Train a collective encoder model",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument('--config', '-c', required=True, type=str,
                            help='Path to configuration YAML file')
        parser.add_argument('--debug', action='store_true',
                            help='Run in debug mode with reduced dataset and epochs')

        args = parser.parse_args()

        # Convert back to format expected by engine
        sys.argv = ['collective-encoder-train', '--config', args.config]
        if args.debug:
            sys.argv.append('--debug')

        # Run the engine
        engine_main()

    finally:
        # Restore original argv
        sys.argv = original_argv


def main():
    """Main entry point that dispatches to subcommands."""
    parser = argparse.ArgumentParser(
        description="Collective Encoder: ML framework for molecular dynamics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', '-c', required=True, type=str,
                             help='Path to configuration YAML file')
    train_parser.add_argument('--debug', action='store_true',
                             help='Run in debug mode')

    args = parser.parse_args()

    if args.command == 'train':
        # Reconstruct argv for train command
        sys.argv = ['collective-encoder-train', '--config', args.config]
        if args.debug:
            sys.argv.append('--debug')
        main_train()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()