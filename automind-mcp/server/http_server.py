"""
HTTP API Server para o AutoMind MCP
Expõe as funcionalidades do servidor MCP via endpoints REST
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from tools.recommender import AutoMindRecommender

app = FastAPI(
    title="AutoMind API",
    description="API para recomendação de veículos",
    version="1.0.0"
)

# Configurar CORS para permitir requisições do Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especifique os domínios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar o recomendador
recommender = AutoMindRecommender()


class RecommendRequest(BaseModel):
    prompt: str = Field(..., description="Descrição do perfil de veículo desejado")
    limit: int = Field(default=3, ge=1, le=20, description="Número de recomendações")


class RecommendResponse(BaseModel):
    filters: list[str]
    preferences: dict
    recommendations: list[dict]
    dataset_insights: dict


@app.get("/")
async def root():
    """Endpoint de health check"""
    return {
        "service": "AutoMind MCP HTTP Server",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Verifica se o servidor está saudável"""
    return {"status": "healthy"}


@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend_vehicles(request: RecommendRequest):
    """
    Retorna recomendações de veículos com base no prompt do usuário
    
    Args:
        request: Objeto contendo o prompt e limite de resultados
        
    Returns:
        Objeto com filtros aplicados, preferências detectadas e recomendações
    """
    try:
        result = recommender.recommend(request.prompt, limit=request.limit)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar recomendação: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
