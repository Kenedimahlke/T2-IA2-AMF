CREATE TABLE IF NOT EXISTS vehicles (
    marca TEXT,
    modelo TEXT,
    ano INTEGER,
    km INTEGER,
    preco_numerico DOUBLE PRECISION,
    consumo_urbano_km_l DOUBLE PRECISION,
    consumo_estrada_km_l DOUBLE PRECISION,
    custo_manutencao_anual INTEGER,
    confiabilidade DOUBLE PRECISION,
    tipo_veiculo_classificado TEXT,
    segmento TEXT,
    codigoFipe TEXT,
    mesReferencia INTEGER,
    anoReferencia INTEGER
);

COPY vehicles (
    marca,
    modelo,
    ano,
    km,
    preco_numerico,
    consumo_urbano_km_l,
    consumo_estrada_km_l,
    custo_manutencao_anual,
    confiabilidade,
    tipo_veiculo_classificado,
    segmento,
    codigoFipe,
    mesReferencia,
    anoReferencia
)
FROM '/seed/vehicles_final.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',');

CREATE INDEX IF NOT EXISTS idx_vehicles_price ON vehicles (preco_numerico);
CREATE INDEX IF NOT EXISTS idx_vehicles_year ON vehicles (ano);
CREATE INDEX IF NOT EXISTS idx_vehicles_type ON vehicles (tipo_veiculo_classificado);

ANALYZE vehicles;
