"""
AutoMind - Setup
Baixa o dataset do Kaggle e prepara o ambiente
"""
import os
import sys


def check_kagglehub():
    """Verifica se kagglehub está instalado"""
    try:
        import kagglehub
        return True
    except ImportError:
        return False


def install_kagglehub():
    """Instala kagglehub"""
    print("\nInstalando kagglehub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
    print("✓ kagglehub instalado com sucesso!")


def download_dataset():
    """Baixa o dataset do Kaggle"""
    import kagglehub
    import shutil
    
    print("\nBaixando dataset da Tabela FIPE do Kaggle...")
    print("(Isso pode levar alguns minutos...)")
    
    # Download do dataset
    path = kagglehub.dataset_download("franckepeixoto/tabela-fipe")
    
    # Copiar para data/raw/
    os.makedirs('data/raw', exist_ok=True)
    
    csv_file = os.path.join(path, 'tabela-fipe-historico-precos.csv')
    destination = 'data/raw/tabela-fipe-historico-precos.csv'
    
    print(f"\nCopiando dataset para {destination}...")
    shutil.copy2(csv_file, destination)
    
    # Verificar tamanho
    size_mb = os.path.getsize(destination) / (1024 * 1024)
    print(f"✓ Dataset baixado com sucesso! ({size_mb:.2f} MB)")


def main():
    print("=" * 70)
    print("AutoMind - Setup do Projeto")
    print("=" * 70)
    
    # Verificar se dataset já existe
    if os.path.exists('data/raw/tabela-fipe-historico-precos.csv'):
        print("\n✓ Dataset já existe em data/raw/")
        print("\nSe deseja baixar novamente, delete o arquivo e execute este script.")
        return
    
    # Verificar kagglehub
    if not check_kagglehub():
        print("\n⚠ kagglehub não encontrado")
        install_kagglehub()
    
    # Baixar dataset
    try:
        download_dataset()
        
        print("\n" + "=" * 70)
        print("Setup concluído com sucesso!")
        print("=" * 70)
        print("\nPróximos passos:")
        print("  1. Execute: python main.py")
        print("  2. Aguarde o processamento (pode levar alguns minutos)")
        print("  3. Dataset final estará em: data/processed/vehicles_final.csv")
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"\n Erro ao baixar dataset: {e}")
        print("\nVocê pode baixar manualmente de:")
        print("https://www.kaggle.com/datasets/franckepeixoto/tabela-fipe")
        print("E salvar em: data/raw/tabela-fipe-historico-precos.csv")
        sys.exit(1)


if __name__ == '__main__':
    main()
