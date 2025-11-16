"""
Integração simplificada do modelo MLP de scoring.

Sistema usa regex para parsing e apenas MLP para scoring de recomendações.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MLPIntegration:
    """
    Gerencia integração do modelo MLP de scoring.
    
    Carrega apenas o RecommendationScorer MLP para ranking de veículos.
    """
    
    def __init__(self):
        self.recommendation_scorer = None
        self._load_models()
    
    def _load_models(self):
        """Carrega modelo MLP de scoring se disponível."""
        try:
            from ml.recommendation_scorer import RecommendationScorer
            self.recommendation_scorer = RecommendationScorer()
            if self.recommendation_scorer.mlp is not None:
                logger.info("✅ Recommendation Scorer MLP carregado")
        except Exception as e:
            logger.warning(f"⚠️ Recommendation Scorer não disponível: {e}")
    
    def has_recommendation_scorer(self) -> bool:
        """Verifica se o scorer está disponível."""
        return self.recommendation_scorer is not None and self.recommendation_scorer.mlp is not None


# Instância global singleton
_mlp_integration: Optional[MLPIntegration] = None


def get_mlp_integration() -> MLPIntegration:
    """Retorna instância singleton da integração MLP."""
    global _mlp_integration
    if _mlp_integration is None:
        _mlp_integration = MLPIntegration()
    return _mlp_integration
