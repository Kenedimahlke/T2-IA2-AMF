"""
AutoMind Data Processor
Processa dataset FIPE do Kaggle e enriquece com dados calculados
"""
import pandas as pd
from src.extractors.enrichment_extractor import EnrichmentExtractor
from src.utils.data_merger import DataMerger


def main():
    print("=" * 70)
    print("AutoMind - Processamento de Dados FIPE")
    print("=" * 70)
    
    # Carregar dataset
    print("\n[1/4] Carregando dataset...")
    df = pd.read_csv('data/raw/tabela-fipe-historico-precos.csv')
    print(f"      {len(df):,} registros carregados")
    
    # Enriquecer dados
    print("\n[2/4] Enriquecendo veiculos com dados calculados...")
    extractor = EnrichmentExtractor()
    enriched = []
    
    for idx, vehicle in enumerate(df.to_dict('records')):
        enriched.append(extractor.enrich_vehicle(vehicle))
        if (idx + 1) % 10000 == 0:
            print(f"      Progresso: {idx + 1:,}/{len(df):,}")
    
    print(f"      {len(enriched):,} veiculos enriquecidos")
    
    # Limpar e organizar
    print("\n[3/4] Removendo duplicatas e organizando...")
    merger = DataMerger()
    df_final = merger.merge_and_clean(enriched)
    print(f"      {len(df_final):,} veiculos unicos")
    
    # Salvar
    print("\n[4/4] Salvando dataset final...")
    output_file = 'data/processed/vehicles_final.csv'
    df_final.to_csv(output_file, index=False)
    print(f"      Salvo em: {output_file}")
    
    # Resumo
    print("\n" + "=" * 70)
    print("RESUMO DO DATASET")
    print("=" * 70)
    print(f"Veiculos: {len(df_final):,}")
    print(f"Marcas: {df_final['marca'].nunique()}")
    print(f"Modelos: {df_final['modelo'].nunique()}")
    print(f"Anos: {df_final['ano'].min()} a {df_final['ano'].max()}")
    print(f"Precos: R$ {df_final['preco_numerico'].min():,.2f} a R$ {df_final['preco_numerico'].max():,.2f}")
    
    print("\nTop 5 marcas:")
    for marca, count in df_final['marca'].value_counts().head(5).items():
        print(f"  - {marca}: {count} modelos")
    
    print("\n" + "=" * 70)
    print("Processamento concluido com sucesso!")
    print("=" * 70)


if __name__ == '__main__':
    main()
