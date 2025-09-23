# Previsão de Satisfação de Clientes - Random Forest

Projeto desenvolvido para a disciplina de Inteligência Artificial II do curso de Bacharelado em Sistemas de Informação da Antônio Meneghetti Faculdade.

## Sobre o Projeto

Este projeto implementa um modelo de Random Forest para prever a satisfação de clientes em um marketplace brasileiro (Olist), permitindo intervenções proativas para melhorar a experiência do cliente.

## Dataset

Utilizado o Brazilian E-Commerce Public Dataset by Olist, disponibilizado publicamente na plataforma Kaggle.

## Principais Resultados

- Accuracy: 0.8023
- F1-score: 0.8812
- ROC-AUC: 0.6874

O modelo identificou os seguintes fatores como principais preditores da satisfação:
1. Atraso na entrega (0.246)
2. Tempo total de entrega (0.196)
3. Proporção entre frete e valor do produto (0.098)

## Como Executar

### Requisitos
- Python 3.8+
- Bibliotecas necessárias:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - shap

### Instalação de Dependências
```bash
pip install pandas numpy scikit-learn matplotlib shap
```

### Execução
```bash
python randomForest.py
```

## Autor

- Kenedi João Mahlke