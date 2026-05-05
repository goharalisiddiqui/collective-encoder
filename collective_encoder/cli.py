#!/usr/bin/env python3
"""Command-line interface for the collective encoder package."""

import argparse
import sys


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--config', '-c', required=True, type=str,
                        help='Path to configuration YAML file')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')


def _run_train(args: argparse.Namespace) -> None:
    from collective_encoder.trainer import main as engine_main
    original_argv = sys.argv.copy()
    try:
        sys.argv = ['collective-encoder-train', '--config', args.config]
        if args.debug:
            sys.argv.append('--debug')
        engine_main()
    finally:
        sys.argv = original_argv


def _run_prepare_dmod(args: argparse.Namespace) -> None:
    from collective_encoder.prepare_dmod import main as prepare_dmod_main
    original_argv = sys.argv.copy()
    try:
        sys.argv = ['collective-encoder-prepare_dmod', '--config', args.config]
        if args.debug:
            sys.argv.append('--debug')
        prepare_dmod_main()
    finally:
        sys.argv = original_argv

def _run_test(args: argparse.Namespace) -> None:
    from collective_encoder.tester import main as tester_main
    original_argv = sys.argv.copy()
    try:
        sys.argv = ['collective-encoder-test', '--config', args.config]
        if args.debug:
            sys.argv.append('--debug')
        tester_main()
    finally:
        sys.argv = original_argv


def main_train() -> None:
    """Entry point for the collective-encoder-train command."""
    parser = argparse.ArgumentParser(
        description="Train a collective encoder model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _add_common_args(parser)
    _run_train(parser.parse_args())


def main_prepare_dmod() -> None:
    """Entry point for the collective-encoder-prepare_dmod command."""
    parser = argparse.ArgumentParser(
        description="Prepare data module for collective encoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _add_common_args(parser)
    _run_prepare_dmod(parser.parse_args())

def main_test() -> None:
    """Entry point for the collective-encoder-test command."""
    parser = argparse.ArgumentParser(
        description="Test a collective encoder model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _add_common_args(parser)
    _run_test(parser.parse_args())


def main() -> None:
    """Main entry point that dispatches to subcommands."""
    parser = argparse.ArgumentParser(
        description="Collective Encoder: ML framework for molecular dynamics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    train_parser = subparsers.add_parser('train', help='Train a model')
    _add_common_args(train_parser)
    train_parser.set_defaults(func=_run_train)

    prepare_parser = subparsers.add_parser('prepare', help='Prepare data module')
    _add_common_args(prepare_parser)
    prepare_parser.set_defaults(func=_run_prepare_dmod)
    
    test_parser = subparsers.add_parser('test', help='Test a model')
    _add_common_args(test_parser)
    test_parser.set_defaults(func=_run_test)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
