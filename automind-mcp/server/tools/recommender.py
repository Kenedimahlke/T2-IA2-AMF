from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .dataset_loader import VehicleDataset, get_dataset
from .preference_parser import UserPreferences, parse_preferences


@dataclass
class Recommendation:
    title: str
    price: float
    year: int
    km: int
    segmento: str
    tipo: str
    consumo: float
    confiabilidade: float
    manutencao: float
    score: float
    rationale: str

    def to_dict(self) -> Dict:
        return {
            "titulo": self.title,
            "preco": round(self.price, 2),
            "ano": int(self.year),
            "km": int(self.km),
            "segmento": self.segmento,
            "tipo": self.tipo,
            "consumo_medio_km_l": round(self.consumo, 1) if not np.isnan(self.consumo) else None,
            "confiabilidade": round(self.confiabilidade, 1),
            "custo_manutencao": int(self.manutencao),
            "score": round(float(self.score), 3),
            "porque_se_destaca": self.rationale,
        }


class AutoMindRecommender:
    def __init__(self, dataset_path: Optional[str] = None) -> None:
        self.dataset = get_dataset(dataset_path)

    def _apply_sanity_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove dados irrealistas do dataset.
        Aplica regras de validação baseadas em lógica de mercado.
        """
        filtered = df.copy()
        
        # Regra: preço mínimo por ano (evita Corolla 2022 por R$30k)
        # Carros novos (2020+) não podem custar menos que R$40k
        # Carros 2015-2019 não podem custar menos que R$20k
        # Carros 2010-2014 não podem custar menos que R$15k
        current_year = 2022  # Atualizar conforme necessário
        
        def get_min_realistic_price(year):
            if year >= 2020:
                return 40000
            elif year >= 2015:
                return 20000
            elif year >= 2010:
                return 15000
            elif year >= 2005:
                return 10000
            else:
                return 5000
        
        filtered['min_realistic_price'] = filtered['ano'].apply(get_min_realistic_price)
        filtered = filtered[filtered['preco_numerico'] >= filtered['min_realistic_price']]
        filtered = filtered.drop(columns=['min_realistic_price'])
        
        # Remover outliers extremos de preço (>R$ 1M para carros populares)
        filtered = filtered[
            (filtered['preco_numerico'] <= 1000000) | 
            (filtered['segmento'].isin(['Premium', 'Luxo']))
        ]
        
        return filtered

    def recommend(self, prompt: str, limit: int = 3) -> Dict:
        prefs = parse_preferences(prompt)
        df = self.dataset.data.copy()
        
        # Filtro de sanidade: remover preços irrealistas baseado no ano
        df = self._apply_sanity_filters(df)
        
        applied_filters: List[str] = []

        if prefs.max_price:
            threshold = prefs.max_price * (1 + prefs.budget_flex)
            df = df[df["preco_numerico"] <= threshold]
            applied_filters.append(f"preço <= R$ {threshold:,.0f}")

        if prefs.min_year:
            df = df[df["ano"] >= prefs.min_year]
            applied_filters.append(f"ano >= {prefs.min_year}")
        
        if prefs.max_year:
            df = df[df["ano"] <= prefs.max_year]
            applied_filters.append(f"ano <= {prefs.max_year}")
        
        if prefs.preferred_types:
            df = df[df["tipo_veiculo_classificado"].isin(prefs.preferred_types)]
            applied_filters.append(
                "tipos: " + ", ".join(sorted(prefs.preferred_types))
            )
            
            # Se pediu sedan, exclui wagons/peruas que podem estar mal classificadas
            if "Sedan" in prefs.preferred_types:
                wagon_keywords = ["fielder", "sw", "variant", "touring", "sportback", "wagon", "perua"]
                for keyword in wagon_keywords:
                    df = df[~df["modelo"].str.lower().str.contains(keyword, na=False)]
                applied_filters.append("excluindo wagons/peruas")
        
        # Filtro adicional para economia - APÓS filtros de tipo
        # Só aplica se ainda houver resultados suficientes
        if "economia" in prefs.priorities and prefs.priorities["economia"] > 0.3 and len(df) > 10:
            # Exclui modelos muito grandes/pouco econômicos
            uneconomical_models = ["fusion", "mondeo", "accord v6", "camry v6"]
            for model in uneconomical_models:
                df = df[~df["modelo"].str.lower().str.contains(model, na=False, regex=False)]
            
            # Filtra por consumo apenas se ainda houver bastante opções
            consumo_threshold = df["consumo_medio"].quantile(0.3)
            if not pd.isna(consumo_threshold) and len(df) > 20:
                df_economico = df[df["consumo_medio"] >= consumo_threshold]
                if len(df_economico) >= 5:
                    df = df_economico
                    applied_filters.append("priorizando economia de combustível")

        if prefs.preferred_brands:
            df_brand = df[df["marca"].isin(prefs.preferred_brands)]
            if not df_brand.empty:
                df = df_brand
                applied_filters.append(
                    "marcas: " + ", ".join(sorted(prefs.preferred_brands))
                )

        if df.empty:
            # Tenta relaxar filtros gradualmente
            df = self.dataset.data.copy()
            df = self._apply_sanity_filters(df)
            
            if prefs.max_price:
                threshold = prefs.max_price * 1.2  # Aceita 20% acima
                df = df[df["preco_numerico"] <= threshold]
            
            if prefs.preferred_types:
                df = df[df["tipo_veiculo_classificado"].isin(prefs.preferred_types)]
            
            if df.empty:
                df = self.dataset.data.nsmallest(500, "preco_numerico")
                applied_filters.append(
                    "⚠ Nenhum veículo atendeu aos filtros; mostrando opções alternativas"
                )
            else:
                applied_filters.append(
                    "⚠ Filtros relaxados para encontrar opções disponíveis"
                )
            
            df = df.nsmallest(500, "preco_numerico")

        ranked = self._score(df, prefs)
        
        # Diversificar por marca e modelo para evitar resultados repetitivos
        top_rows = self._diversify_recommendations(ranked, limit)

        recommendations = [self._row_to_recommendation(row, prefs) for _, row in top_rows.iterrows()]

        return {
            "filters": applied_filters,
            "preferences": prefs.to_dict(),
            "recommendations": [rec.to_dict() for rec in recommendations],
            "dataset_insights": {
                "amostra": len(top_rows),
                "media_preco": round(float(top_rows["preco_numerico"].mean()), 2),
                "media_consumo": round(float(top_rows["consumo_medio"].mean()), 2),
            },
        }

    def _score(self, df: pd.DataFrame, prefs: UserPreferences) -> pd.DataFrame:
        """
        Calcula scores para os veículos.
        Tenta usar MLP scorer se disponível, caso contrário usa método tradicional.
        """
        scored = df.copy()
        
        # Tentar usar MLP scorer primeiro
        try:
            from ml.integration import get_mlp_integration
            
            mlp = get_mlp_integration()
            if mlp.has_recommendation_scorer():
                # Preparar preferências para o MLP
                prefs_dict = {
                    'wants_economical': 'economia' in prefs.priorities,
                    'budget_max': prefs.max_price or 100000,
                    'vehicle_types': list(prefs.preferred_types) if prefs.preferred_types else []
                }
                
                # Usar MLP para scoring
                mlp_scores = mlp.recommendation_scorer.score_vehicles(scored, prefs_dict)
                scored["score"] = mlp_scores
                
                # Ainda aplicar bonus de ano específico
                if prefs.preferred_year:
                    year_diff = (scored["ano"] - prefs.preferred_year).abs()
                    year_bonus = (1 - (year_diff / 10)).clip(lower=0)
                    scored["score"] = scored["score"] + (year_bonus * 5)  # Bonus adicional
                
                return scored.sort_values("score", ascending=False)
        except Exception as e:
            # Fallback para método tradicional
            import logging
            logging.warning(f"MLP scoring failed, using traditional method: {e}")
        
        # FALLBACK: Método tradicional (código original)
        scored["norm_price"] = self._price_alignment_score(scored["preco_numerico"], prefs)
        scored["norm_consumo"] = self._normalize(scored["consumo_medio"], default=0.5)
        scored["norm_confiabilidade"] = self._normalize(scored["confiabilidade"], default=0.5)
        scored["norm_manutencao"] = self._normalize(
            scored["custo_manutencao_anual"], invert=True, default=0.5
        )
        scored["norm_idade"] = self._normalize(scored["idade"], invert=True)
        
        # Bonus para ano preferido específico
        if prefs.preferred_year:
            year_diff = (scored["ano"] - prefs.preferred_year).abs()
            scored["year_bonus"] = (1 - (year_diff / 10)).clip(lower=0)  # Penaliza diferenças grandes
        else:
            scored["year_bonus"] = 0

        weights = self._build_weights(prefs)
        scored["score"] = (
            weights["preco"] * scored["norm_price"]
            + weights["economia"] * scored["norm_consumo"]
            + weights["confiabilidade"] * scored["norm_confiabilidade"]
            + weights["manutencao"] * scored["norm_manutencao"]
            + weights["idade"] * scored["norm_idade"]
            + 0.15 * scored["year_bonus"]  # Bonus para ano específico
        )

        return scored.sort_values("score", ascending=False)

    def _diversify_recommendations(self, df: pd.DataFrame, limit: int) -> pd.DataFrame:
        """
        Diversifica recomendações para evitar mostrar apenas variações do mesmo modelo.
        Prioriza diferentes marcas e modelos, mantendo os melhores scores.
        """
        if len(df) <= limit:
            return df.head(limit)
        
        selected = []
        seen_models = set()
        
        def get_base_model(model_name: str) -> str:
            """Extrai o nome base do modelo (primeira palavra significativa)"""
            if not model_name:
                return ""
            import re
            # Remove números e especificações técnicas, pega a primeira palavra
            cleaned = re.sub(r'\d+\.?\d*', '', model_name)  # Remove todos os números
            cleaned = re.sub(r'\b(flex|aut\.?|mec\.?|diesel|gasolina|4x4|4x2|awd|fwd|mpfi|16v|24v|turbo|tsi|crde|tb)\b', '', cleaned, flags=re.IGNORECASE)
            words = [w.strip() for w in cleaned.split() if w.strip() and len(w.strip()) > 1]
            return words[0] if words else model_name.split()[0]
        
        # Estratégia: prioriza diversidade de modelos base
        for _, row in df.iterrows():
            if len(selected) >= limit:
                break
            
            # Cria chave única por marca + modelo base
            base_model = get_base_model(row.get('modelo', ''))
            model_key = f"{row.get('marca', '')}_{base_model}".lower().strip()
            
            # Adiciona apenas se for um modelo base diferente
            if model_key not in seen_models:
                selected.append(row)
                seen_models.add(model_key)
        
        # Se não conseguimos o limite, completa com o que tiver (evitando duplicatas exatas)
        if len(selected) < limit:
            for _, row in df.iterrows():
                if len(selected) >= limit:
                    break
                    
                is_duplicate = any(
                    (r.get("marca") == row.get("marca") and 
                     r.get("modelo") == row.get("modelo") and 
                     r.get("ano") == row.get("ano"))
                    for r in selected
                )
                
                if not is_duplicate:
                    selected.append(row)
        
        return pd.DataFrame(selected).head(limit)

    @staticmethod
    def _normalize(series: pd.Series, invert: bool = False, default: float = 0.0) -> pd.Series:
        valid = series.dropna()
        if valid.empty:
            return pd.Series(default, index=series.index)
        min_val, max_val = valid.min(), valid.max()
        if min_val == max_val:
            normalized = pd.Series(0.5, index=series.index)
        else:
            normalized = (series - min_val) / (max_val - min_val)
        if invert:
            normalized = 1 - normalized
        return normalized.fillna(default)

    def _price_alignment_score(self, series: pd.Series, prefs: UserPreferences) -> pd.Series:
        if prefs.max_price:
            target = max(prefs.max_price * 0.85, 1)
            ceiling = prefs.max_price * (1 + prefs.budget_flex)
            relative_diff = (series - target).abs() / target
            score = 1 - relative_diff
            score = score.clip(lower=0)
            score = score.where(series <= ceiling, other=0)
            return score.fillna(0.5)

        return self._normalize(series, invert=True)

    @staticmethod
    def _build_weights(prefs: UserPreferences) -> Dict[str, float]:
        weights = {
            "preco": 0.3,
            "economia": 0.2,
            "confiabilidade": 0.2,
            "manutencao": 0.15,
            "idade": 0.15,
        }

        mapping = {
            "custo": "preco",
            "economia": "economia",
            "manutencao": "manutencao",
            "confiabilidade": "confiabilidade",
            "tecnologia": "idade",
        }

        for priority, value in prefs.priorities.items():
            key = mapping.get(priority)
            if key:
                weights[key] += 0.2 * value

        total = sum(weights.values())
        return {key: weight / total for key, weight in weights.items()}

    def _row_to_recommendation(self, row: pd.Series, prefs: UserPreferences) -> Recommendation:
        rationale_parts = []
        if prefs.max_price:
            ratio = row["preco_numerico"] / prefs.max_price
            if ratio <= 1:
                rationale_parts.append("dentro do orçamento")
            else:
                rationale_parts.append("ligeiramente acima do orçamento, mas competitivo")

        if row.get("consumo_medio"):
            rationale_parts.append(
                f"consumo médio estimado de {row['consumo_medio']:.1f} km/l"
            )
        if row.get("confiabilidade"):
            rationale_parts.append(
                f"índice de confiabilidade {row['confiabilidade']:.1f}/10"
            )
        if row.get("custo_manutencao_anual"):
            rationale_parts.append(
                f"custo anual aproximado de R$ {row['custo_manutencao_anual']:,.0f}"
            )

        rationale = ", ".join(rationale_parts) if rationale_parts else "bom equilíbrio geral"

        return Recommendation(
            title=f"{row['marca']} {row['modelo']} ({row['ano']})",
            price=float(row["preco_numerico"]),
            year=int(row["ano"]),
            km=int(row["km"]),
            segmento=row.get("segmento", "Desconhecido"),
            tipo=row.get("tipo_veiculo_classificado", "N/A"),
            consumo=float(row.get("consumo_medio", np.nan)),
            confiabilidade=float(row.get("confiabilidade", np.nan)),
            manutencao=float(row.get("custo_manutencao_anual", np.nan)),
            score=float(row["score"]),
            rationale=rationale,
        )


if __name__ == "__main__":
    prompt = (
        "Tenho até R$ 90 mil para comprar um carro econômico para uso na cidade, "
        "com baixa manutenção e boa confiabilidade."
    )
    recommender = AutoMindRecommender()
    resultado = recommender.recommend(prompt, limit=3)
    for carro in resultado["recommendations"]:
        print(carro["titulo"], "- score", carro["score"])
        print(carro["porque_se_destaca"])
        print("-")
