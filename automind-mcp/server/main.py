from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

from tools.recommender import AutoMindRecommender

server = FastMCP("automind")
recommender = AutoMindRecommender()


def _format_response(payload: dict) -> str:
    lines: list[str] = []
    lines.append("AutoMind - Sugestões de veículos")
    lines.append("")

    if payload.get("filters"):
        lines.append("Filtros aplicados: " + "; ".join(payload["filters"]))
        lines.append("")

    for index, item in enumerate(payload["recommendations"], start=1):
        lines.append(f"{index}. {item['titulo']}")
        lines.append(
            f"   Preço: R$ {item['preco']:,.0f} | Ano: {item['ano']} | Km estimado: {item['km']:,}"
        )
        lines.append(
            f"   Consumo: {item['consumo_medio_km_l']} km/l | Confiabilidade: {item['confiabilidade']}"
        )
        lines.append(
            f"   Manutenção: R$ {item['custo_manutencao']:,}/ano | Score: {item['score']}"
        )
        lines.append(f"   Destaque: {item['porque_se_destaca']}")
        lines.append("")

    lines.append("Resumo da amostra: " + json.dumps(payload["dataset_insights"], ensure_ascii=False))
    return "\n".join(lines)


@server.tool()
def recomendar_carros(
    prompt: Annotated[str, "Descrição livre do perfil desejado"],
    quantidade: Annotated[int, "Número máximo de resultados"] = 3,
) -> list[TextContent]:
    """Retorna recomendações de veículos com base no prompt enviado."""

    resultado = recommender.recommend(prompt, limit=quantidade)
    formatted = _format_response(resultado)
    return [TextContent(type="text", text=formatted)]


if __name__ == "__main__":
    server.run()
