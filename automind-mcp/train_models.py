#!/usr/bin/env python3
"""
Script para treinar modelo MLP de scoring do AutoMind.

Treina apenas o Recommendation Scorer (sistema usa regex para parsing).
"""
import sys
from pathlib import Path

# Adicionar diretório server ao path
sys.path.insert(0, str(Path(__file__).parent / "server"))

from ml.recommendation_scorer import train_recommendation_scorer


def main():
    """Treina o modelo de scoring."""
    print("=" * 80)
    print("🚗 AUTOMIND - TREINAMENTO DO RECOMMENDATION SCORER")
    print("=" * 80)
    
    try:
        print("\n\n🎯 TREINANDO SCORER DE RECOMENDAÇÕES\n")
        print("-" * 80)
        recommendation_scorer = train_recommendation_scorer()
        print("\n✅ Scorer de recomendações treinado com sucesso!")
        
        print("\n\n" + "=" * 80)
        print("✅ MODELO TREINADO COM SUCESSO!")
        print("=" * 80)
        print("\n📁 Modelo salvo em: server/ml/models/")
        print("   - recommendation_scorer.pkl")
        print("\n🚀 O modelo está pronto para uso no sistema AutoMind!")
        print("\n💡 Sistema usa REGEX para parsing e MLP apenas para scoring!")
        
    except Exception as e:
        print(f"\n\n❌ ERRO no treinamento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
