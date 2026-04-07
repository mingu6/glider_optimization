import sys
from pathlib import Path
from typing import Optional

from glider_optimization.config import load_config
from glider_optimization.logger import setup_logging

from argparse import ArgumentParser

def build_parser() -> ArgumentParser:
    p = ArgumentParser(prog="bilevel-airfoil")
    p.add_argument("--config", "-c", type=Path, required=True)
    p.add_argument("--run-name", "-n", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--debug", action="store_true")
    return p

def parse_args(args: Optional[list] = None):
    return build_parser().parse_args(args)

def _apply_overrides(cfg, args):
    if args.device is not None:
        cfg.run.device = args.device
    if args.seed is not None:
        cfg.run.seed = args.seed
    if args.run_name is not None:
        cfg.io.run_name = args.run_name
    cfg.io.debug = bool(args.debug)
    return cfg

def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    cfg = _apply_overrides(cfg, args)
    setup_logging(cfg.io)
    
    from glider_optimization.runner import Runner
    runner = Runner(cfg)
    try:
        runner.run()
    except KeyboardInterrupt:
        runner.checkpoint_on_interrupt()
        return 130
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
