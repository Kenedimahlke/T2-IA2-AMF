from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

VEHICLE_KEYWORDS = {
    "SUV": {"suv", "tracker", "compass", "creta", "kicks", "renegade", "t-cross"},
    "Sedan": {"sedan", "sedã", "sedans", "civic", "versa", "virtus", "jetta", "city", "fit sedan"},
    "Hatchback": {"hatch", "hb20", "onix", "argo", "gol", "polo", "fit", "yaris"},
    "Wagon": {"wagon", "perua", "sw", "fielder", "variant", "parati", "spacefox"},
    "Picape": {"picape", "pickup", "s10", "hilux", "amarok", "ranger", "strada", "toro"},
    "Esportivo": {"esportivo", "cupê", "coupe", "mustang", "camaro", "gti"},
}

USO_KEYWORDS = {
    "cidade": {"cidade", "urbano", "trânsito", "dia a dia"},
    "estrada": {"viagem", "estrada", "rodovia", "longa distância"},
    "misto": {"misto", "ambos", "híbrido"},
}

PRIORITY_KEYWORDS = {
    "custo": {"preço", "barato", "custo", "acessível"},
    "economia": {"economia", "consumo", "econômico", "economico", "gasta pouco", "baixo consumo"},
    "manutencao": {"manutenção", "manutencao", "peças", "pecas"},
    "confiabilidade": {"confiável", "confiavel", "durabilidade", "segurança", "seguranca"},
    "potencia": {"potência", "potente", "desempenho"},
    "tecnologia": {"tecnologia", "conectividade", "multimídia", "moderno", "moderna", "atual", "recente", "novo", "nova", "novos", "mais novo"},
}

RECENCY_KEYWORDS = [
    "mais novo",
    "o mais novo",
    "novo possível",
    "novo possivel",
    "recente",
    "moderno",
    "atual",
    "atualizado",
    "modelo novo",
    "seminovo",
    "zero km",
    "0km",
    "0 km",
]

BRAND_NAMES = {
    "Toyota",
    "Honda",
    "Hyundai",
    "Chevrolet",
    "Volkswagen",
    "Fiat",
    "Jeep",
    "Nissan",
    "Renault",
    "Peugeot",
    "Citroën",
    "Ford",
    "Kia",
    "BMW",
    "Mercedes",
    "Audi",
}

PRICE_PATTERN = re.compile(r"(?:r\$\s*)?(\d{1,3}(?:[\.\s]\d{3})*(?:,\d{1,2})?|\d+(?:,\d{1,2})?)", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"(20\d{2}|19\d{2})")
SPECIFIC_YEAR_PATTERN = re.compile(r"\b(?:ano|model[oa])\s+(20\d{2}|19\d{2})\b", re.IGNORECASE)


@dataclass
class UserPreferences:
    original_prompt: str
    max_price: Optional[float] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None  # Ano máximo (para "até ano X")
    preferred_year: Optional[int] = None  # Ano específico preferido
    max_km: Optional[int] = None
    preferred_types: Set[str] = field(default_factory=set)
    usage_profile: str = "misto"
    priorities: Dict[str, float] = field(default_factory=dict)
    preferred_brands: Set[str] = field(default_factory=set)
    banned_brands: Set[str] = field(default_factory=set)
    budget_flex: float = 0.1
    language: str = "pt-BR"

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload["preferred_types"] = sorted(self.preferred_types)
        payload["preferred_brands"] = sorted(self.preferred_brands)
        payload["banned_brands"] = sorted(self.banned_brands)
        return payload


def _parse_price(text: str) -> Optional[float]:
    matches = PRICE_PATTERN.findall(text)
    if not matches:
        return None

    # Usa o maior valor encontrado presumindo que seja o teto do orçamento
    value = matches[-1]
    normalized = value.replace(".", "").replace(" ", "").replace(",", ".")
    try:
        number = float(normalized)
    except ValueError:
        return None

    # Valores abaixo de 1.000 provavelmente foram descritos em "mil"
    if number < 1000:
        number *= 1000

    return number


def _parse_year(text: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Retorna (preferred_year, min_year, max_year)
    - preferred_year: quando menciona 'ano 2009' (quer exatamente esse ano ou próximo)
    - min_year: quando menciona 'a partir de 2009'
    - max_year: quando menciona 'até ano 2015'
    """
    preferred_year = None
    min_year = None
    max_year = None
    
    # Checa se é ano máximo: "até ano X", "no máximo ano X"
    max_year_match = re.search(r"(?:até|no máximo|antes de)\s+(?:ano|model[oa])?\s*(20\d{2}|19\d{2})", text, re.IGNORECASE)
    if max_year_match:
        max_year = int(max_year_match.group(1))
        return None, None, max_year
    
    # Checa se é ano mínimo: "a partir de X", "desde X"
    min_year_match = re.search(r"(?:a partir de|após|acima de|desde)\s+(?:ano|model[ola])?\s*(20\d{2}|19\d{2})", text, re.IGNORECASE)
    if min_year_match:
        min_year = int(min_year_match.group(1))
        return None, min_year, None
    
    # Checa se é ano específico: "ano 2009", "modelo 2009"
    specific_match = SPECIFIC_YEAR_PATTERN.search(text)
    if specific_match:
        preferred_year = int(specific_match.group(1))
        return preferred_year, None, None
    
    # Fallback: pega qualquer ano mencionado
    years = [int(match) for match in YEAR_PATTERN.findall(text)]
    years = [year for year in years if 1998 <= year <= 2025]
    if years:
        # Se só menciona um ano, assume que é específico
        year_val = years[0] if len(years) == 1 else max(years)
        is_specific = len(years) == 1
        if is_specific:
            return year_val, None, None
        else:
            return None, year_val, None
    
    return None, None, None


def _infer_recent_year(text: str) -> Optional[int]:
    lowered = text.lower()
    if any(keyword in lowered for keyword in RECENCY_KEYWORDS):
        current_year = datetime.now().year
        # Evita restringir demais caso o dataset tenha anos mais antigos
        baseline = max(2010, current_year - 8)
        return baseline
    return None


def _deduce_usage(text: str) -> str:
    text_lower = text.lower()
    for usage, keywords in USO_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return usage
    return "misto"


def _detect_types(text: str) -> Set[str]:
    text_lower = text.lower()
    detected: Set[str] = set()
    for vehicle_type, keywords in VEHICLE_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            detected.add(vehicle_type)
    return detected


def _detect_priorities(text: str) -> Dict[str, float]:
    text_lower = text.lower()
    priorities: Dict[str, float] = {}
    for priority, keywords in PRIORITY_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            priorities[priority] = priorities.get(priority, 0.0) + 1.0
    if not priorities:
        priorities["equilibrio"] = 1.0
    total = sum(priorities.values())
    return {key: value / total for key, value in priorities.items()}


def _detect_brands(text: str) -> Set[str]:
    detected = set()
    for brand in BRAND_NAMES:
        if brand.lower() in text.lower():
            detected.add(brand)
    return detected


def parse_preferences(prompt: str) -> UserPreferences:
    """
    Extrai preferências do usuário a partir do texto usando regex patterns.
    """
    max_price = _parse_price(prompt)
    preferred_year, min_year, max_year = _parse_year(prompt)
    
    # Se não veio max_year mas veio preferred_year, calcula range
    if preferred_year and not max_year:
        # Usuário quer ano específico, aceita +/- 2 anos
        min_year = preferred_year - 2
        max_year = preferred_year + 2
    
    # Checa se quer carro recente (caso não tenha ano definido)
    if not preferred_year and not min_year and not max_year:
        inferred = _infer_recent_year(prompt)
        if inferred:
            min_year = inferred
    
    usage = _deduce_usage(prompt)
    preferred_types = _detect_types(prompt)
    priorities = _detect_priorities(prompt)
    brands = _detect_brands(prompt)

    return UserPreferences(
        original_prompt=prompt,
        max_price=max_price,
        min_year=min_year,
        max_year=max_year,
        preferred_year=preferred_year,
        usage_profile=usage,
        preferred_types=preferred_types,
        priorities=priorities,
        preferred_brands=brands,
    )
