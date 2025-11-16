from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1] / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.append(str(SERVER_DIR))

from tools.recommender import AutoMindRecommender  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa uma consulta local ao AutoMind")
    parser.add_argument("--prompt", required=True, help="Descrição livre do carro desejado")
    parser.add_argument("--limit", type=int, default=3, help="Quantidade de sugestões")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Caminho alternativo para vehicles_final.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = str(args.dataset) if args.dataset else None
    recommender = AutoMindRecommender(dataset_path)
    result = recommender.recommend(args.prompt, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
