import logging
from pathlib import Path

def setup_logging(io_cfg) -> None:
    log_dir = Path(io_cfg.checkpoint_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"

    logging.basicConfig(
        level=logging.INFO if not io_cfg.debug else logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler()
        ],
        force=True,
    )
