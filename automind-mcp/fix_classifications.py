#!/usr/bin/env python3
import os
from sqlalchemy import create_engine, text

db_url = os.getenv("AUTOMIND_DB_URL", "postgresql://automind:automind@db:5432/automind")
engine = create_engine(db_url)

# IMPORTANTE: Corrigir wagons PRIMEIRO para evitar que "Sedan Touring" seja classificado como Sedan
wagon_query = text("""
    UPDATE vehicles 
    SET tipo_veiculo_classificado = 'Wagon' 
    WHERE LOWER(modelo) LIKE '%touring%' 
       OR LOWER(modelo) LIKE '%fielder%' 
       OR LOWER(modelo) LIKE '%variant%' 
       OR LOWER(modelo) ~ '\\msw\\M'
       OR LOWER(modelo) LIKE '%wagon%'
       OR LOWER(modelo) LIKE '%perua%'
""")

with engine.connect() as conn:
    result = conn.execute(wagon_query)
    conn.commit()
    print(f"✅ Atualizados {result.rowcount} wagons")

# Depois corrigir sedans (mas não sobrescrever wagons)
sedan_query = text("""
    UPDATE vehicles 
    SET tipo_veiculo_classificado = 'Sedan' 
    WHERE (LOWER(modelo) LIKE '%sedan%' 
           OR LOWER(modelo) LIKE '%classic%' 
           OR LOWER(modelo) LIKE '%voyage%' 
           OR LOWER(modelo) LIKE '%logan%' 
           OR LOWER(modelo) LIKE '%prisma%' 
           OR LOWER(modelo) LIKE '%cobalt%')
      AND tipo_veiculo_classificado != 'Wagon'
""")

with engine.connect() as conn:
    result = conn.execute(sedan_query)
    conn.commit()
    print(f"✅ Atualizados {result.rowcount} sedans (sem sobrescrever wagons)")

# Verificar resultado
verify_query = text("""
    SELECT tipo_veiculo_classificado, COUNT(*) as total 
    FROM vehicles 
    GROUP BY tipo_veiculo_classificado 
    ORDER BY total DESC
""")

print("\n📊 Distribuição por tipo:")
with engine.connect() as conn:
    result = conn.execute(verify_query)
    for row in result:
        print(f"  {row[0]}: {row[1]}")

engine.dispose()
