#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app/server')

from tools.dataset_loader import get_dataset

# Limpar cache e recarregar
get_dataset.cache_clear()
df = get_dataset().data

print('=== VERIFICAÇÃO FINAL ===\n')

# Verificar sedans mal classificados como hatch
sedans_as_hatch = df[
    (df['tipo_veiculo_classificado'] == 'Hatchback') & 
    (df['modelo'].str.contains('sedan|classic|voyage|logan|prisma|cobalt', case=False, na=False, regex=True))
]
print(f'❌ Sedans AINDA como Hatch: {len(sedans_as_hatch)}')
if len(sedans_as_hatch) > 0:
    print(sedans_as_hatch[['marca', 'modelo', 'tipo_veiculo_classificado']].drop_duplicates().head(10))

# Verificar wagons mal classificados como sedan
wagons_as_sedan = df[
    (df['tipo_veiculo_classificado'] == 'Sedan') & 
    (df['modelo'].str.contains('touring|fielder|variant| sw |wagon', case=False, na=False, regex=True))
]
print(f'\n❌ Wagons AINDA como Sedan: {len(wagons_as_sedan)}')
if len(wagons_as_sedan) > 0:
    print(wagons_as_sedan[['marca', 'modelo', 'tipo_veiculo_classificado']].drop_duplicates().head(10))

# Mostrar wagons corretamente classificados
print(f'\n✅ Wagons corretamente classificados: {len(df[df["tipo_veiculo_classificado"] == "Wagon"])}')
wagons = df[df['tipo_veiculo_classificado'] == 'Wagon'][['marca', 'modelo']].drop_duplicates().head(20)
print(wagons.to_string(index=False))

# Distribuição final
print('\n📊 Distribuição por tipo:')
print(df['tipo_veiculo_classificado'].value_counts())
