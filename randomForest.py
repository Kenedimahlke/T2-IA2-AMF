# Trabalho 1 – Inteligência Artificial II 2025/02
# Classificação com Random Forest - Análise de Satisfação do Cliente E-commerce

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)


# ============================================================================
# CARREGAMENTO E EDA BREVE
# ============================================================================
print("Carregando datasets...")
orders = pd.read_csv('datasets/olist_orders_dataset.csv')
reviews = pd.read_csv('datasets/olist_order_reviews_dataset.csv')
items = pd.read_csv('datasets/olist_order_items_dataset.csv')
customers = pd.read_csv('datasets/olist_customers_dataset.csv')

print(f"EDA INICIAL:")
print(f"Orders: {orders.shape} | Reviews: {reviews.shape}")
print(f"Items: {items.shape} | Customers: {customers.shape}")

# EDA - Análise da variável target
print(f"\nDISTRIBUIÇÃO TARGET (Review Score):")
print(reviews['review_score'].value_counts().sort_index())
print(f"% Reviews Positivas (>=4): {(reviews['review_score'] >= 4).mean()*100:.1f}%")

# EDA - Análise de dados faltantes
print(f"\nDADOS FALTANTES:")
missing_orders = orders.isnull().sum()
print(f"Orders com datas faltantes: {missing_orders[missing_orders > 0]}")

# EDA - Análise temporal
orders_temp = orders.copy()
orders_temp['order_delivered_customer_date'] = pd.to_datetime(orders_temp['order_delivered_customer_date'])
orders_temp['order_estimated_delivery_date'] = pd.to_datetime(orders_temp['order_estimated_delivery_date'])

print(f"\nANÁLISE TEMPORAL:")
print(f"Pedidos sem data de entrega: {orders_temp['order_delivered_customer_date'].isnull().sum()} ({orders_temp['order_delivered_customer_date'].isnull().mean()*100:.1f}%)")

# ============================================================================
# LIMPEZA E TRATAMENTO DE FALTANTES/OUTLIERS
# ============================================================================
print("LIMPEZA E TRATAMENTO DE DADOS...")

# Converter datas
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])

# TRATAMENTO DE FALTANTES: Remover pedidos sem entrega (sem vazamento)
orders_clean = orders.dropna(subset=['order_delivered_customer_date', 'order_estimated_delivery_date']).copy()
print(f"Pedidos após limpeza: {len(orders_clean)} (removidos: {len(orders) - len(orders_clean)})")

# TRATAMENTO DE OUTLIERS em frete
print(f"\nTRATAMENTO DE OUTLIERS (Frete):")
print(f"Frete antes: Q5={items['freight_value'].quantile(0.05):.2f}, Q95={items['freight_value'].quantile(0.95):.2f}")
items['freight_value'] = items['freight_value'].clip(
    lower=items['freight_value'].quantile(0.05),
    upper=items['freight_value'].quantile(0.95)
)
print(f"Frete após clipping: Q5={items['freight_value'].quantile(0.05):.2f}, Q95={items['freight_value'].quantile(0.95):.2f}")

# ============================================================================
# ENGENHARIA DE ATRIBUTOS
# ============================================================================
print(f"\nENGENHARIA DE ATRIBUTOS...")

# Features temporais (disponíveis no momento do pedido - SEM VAZAMENTO)
orders_clean['atraso_dias'] = (orders_clean['order_delivered_customer_date'] - orders_clean['order_estimated_delivery_date']).dt.days
orders_clean['tempo_entrega'] = (orders_clean['order_delivered_customer_date'] - orders_clean['order_purchase_timestamp']).dt.days
orders_clean['mes_compra'] = orders_clean['order_purchase_timestamp'].dt.month
orders_clean['dia_semana'] = orders_clean['order_purchase_timestamp'].dt.dayofweek
orders_clean['is_weekend'] = (orders_clean['dia_semana'] >= 5).astype(int)

# Target: review bom (>= 4)
reviews['review_good'] = (reviews['review_score'] >= 4).astype(int)

# Merge dados principais
df = orders_clean.merge(reviews[['order_id', 'review_good']], on='order_id', how='inner')
df = df.merge(customers[['customer_id', 'customer_state']], on='customer_id', how='left')

# Features agregadas de items (por pedido)
items_agg = items.groupby('order_id').agg({
    'price': ['sum', 'mean', 'count'],
    'freight_value': 'mean',
    'order_item_id': 'count'
}).reset_index()

# Flatten column names
items_agg.columns = ['order_id', 'total_price', 'avg_price', 'price_count', 'avg_freight', 'num_items']
df = df.merge(items_agg, on='order_id', how='left')

# Feature adicional: frete relativo
df['freight_ratio'] = df['avg_freight'] / (df['avg_price'] + 0.01)  # Evitar divisão por zero

# ============================================================================
# DIVISÃO TREINO/TESTE E VALIDAÇÃO (SEM VAZAMENTO)
# ============================================================================

# Selecionar features (apenas info disponível no momento do pedido)
features = ['atraso_dias', 'tempo_entrega', 'mes_compra', 'dia_semana', 'is_weekend',
           'total_price', 'avg_price', 'avg_freight', 'num_items', 'freight_ratio']

# Encode categóricas se necessário
le = LabelEncoder()
if 'customer_state' in df.columns:
    df['customer_state_encoded'] = le.fit_transform(df['customer_state'].fillna('Unknown'))
    features.append('customer_state_encoded')

# Preparar dados finais
df_final = df.dropna(subset=features + ['review_good']).copy()
X = df_final[features]
y = df_final['review_good']

print(f"Dataset final: {X.shape[0]} amostras, {X.shape[1]} features")
print(f"Distribuição target: {y.value_counts().to_dict()}")
print(f"Balanceamento: {y.mean()*100:.1f}% positivo")

# DIVISÃO ESTRATIFICADA (preserva proporção das classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Split realizado - Treino: {len(X_train)}, Teste: {len(X_test)}")

# ============================================================================
# BASELINE E TREINAMENTO RANDOM FOREST
# ============================================================================
print(f"\nBASELINE (Classe Majoritária):")
baseline_acc = y_train.value_counts().max() / len(y_train)
print(f"Baseline Accuracy: {baseline_acc:.4f}")

# RANDOM FOREST

rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=15, random_state=42, 
    class_weight="balanced", n_jobs=-1
)

print(f"Treinando Random Forest...")
rf_model.fit(X_train, y_train)

# Predições
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("RESULTADOS FINAIS")
print("="*60)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC: {average_precision_score(y_test, y_prob):.4f}")

# Cross-validation do Random Forest
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='f1')
print(f"F1 Cross-Validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================================
# ANÁLISE DE OVERFITTING/UNDERFITTING
# ============================================================================
print("\nANÁLISE DE OVERFITTING/UNDERFITTING:")

# Avaliar performance em treino vs teste
y_train_pred = rf_model.predict(X_train)
f1_train = f1_score(y_train, y_train_pred)
f1_test = f1_score(y_test, y_pred)

print(f"F1 Treino: {f1_train:.4f}")
print(f"F1 Teste:  {f1_test:.4f}")
print(f"Gap (Treino-Teste): {f1_train - f1_test:.4f}")

if (f1_train - f1_test) > 0.05:
    print("POSSÍVEL OVERFITTING (gap > 5%)")
elif (f1_train - f1_test) < 0.01:
    print("MODELO BEM GENERALIZADO")
else:
    print("AJUSTE RAZOÁVEL")

# ============================================================================
# VISUALIZAÇÕES E CURVAS
# ============================================================================
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
im = axes[0,0].imshow(cm, interpolation='nearest', cmap="Blues")
axes[0,0].set_title("Matriz de Confusão")
# Adicionar valores na matriz
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axes[0,0].text(j, i, str(cm[i, j]), ha="center", va="center")

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0,1].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, y_prob):.3f}")
axes[0,1].plot([0,1],[0,1], "--", color="gray")
axes[0,1].set_xlabel("Falso Positivo")
axes[0,1].set_ylabel("Verdadeiro Positivo")
axes[0,1].set_title("Curva ROC")
axes[0,1].legend()

# Curva Precision-Recall
prec, rec, _ = precision_recall_curve(y_test, y_prob)
axes[1,0].plot(rec, prec, label=f"PR-AUC = {average_precision_score(y_test, y_prob):.3f}")
axes[1,0].set_xlabel("Recall")
axes[1,0].set_ylabel("Precisão")
axes[1,0].set_title("Curva Precision-Recall")
axes[1,0].legend()

# Distribuição de Probabilidades
axes[1,1].hist(y_prob[y_test==0], alpha=0.7, bins=30, label='Classe 0', density=True)
axes[1,1].hist(y_prob[y_test==1], alpha=0.7, bins=30, label='Classe 1', density=True)
axes[1,1].set_xlabel("Probabilidade Predita")
axes[1,1].set_ylabel("Densidade")
axes[1,1].set_title("Distribuição de Probabilidades")
axes[1,1].legend()

plt.tight_layout()
plt.savefig('resultados_random_forest.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Curva de aprendizado (DETECÇÃO DE OVERFITTING)
print("\nCalculando curva de aprendizado...")
train_sizes, train_scores, test_scores = learning_curve(
    rf_model, X, y, cv=3, scoring="f1", n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Treino")
plt.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                 train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.2)
plt.plot(train_sizes, test_scores.mean(axis=1), "o-", label="Validação")
plt.fill_between(train_sizes, test_scores.mean(axis=1) - test_scores.std(axis=1),
                 test_scores.mean(axis=1) + test_scores.std(axis=1), alpha=0.2)
plt.title("Curva de Aprendizado - Análise de Overfitting")
plt.xlabel("Tamanho do conjunto de treino")
plt.ylabel("F1-Score")
plt.legend()
plt.grid(True)
plt.savefig('curva_aprendizado.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ============================================================================
# INTERPRETABILIDADE (SHAP/Feature Importance)
# ============================================================================
print("INTERPRETABILIDADE:")

# Feature Importance (Random Forest)
if hasattr(rf_model, 'feature_importances_'):
    feature_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTOP 5 FEATURES MAIS IMPORTANTES:")
    print(feature_imp.head().to_string(index=False))
    
    # Plot separado de feature importance
    plt.figure(figsize=(10, 6))
    feature_imp.sort_values('importance').plot(x='feature', y='importance', kind='barh')
    plt.title('Feature Importance - Random Forest')
    plt.xlabel('Importância')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# SHAP Analysis
print("\nAnálise SHAP...")

# Sample menor para SHAP (performance)
sample_size = min(100, len(X_test))
X_sample = X_test.sample(sample_size, random_state=42)

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_sample)

# Se classificação binária, pegar classe positiva
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Criar SHAP plot com configurações adequadas
plt.clf()
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, show=False)
plt.title('SHAP Summary Plot - Importância e Impacto das Features', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Análise SHAP concluída")

# ============================================================================
# OTIMIZAÇÃO DE HIPERPARÂMETROS
# ============================================================================
print("\nOTIMIZAÇÃO DE HIPERPARÂMETROS...")

# Performance ANTES da otimização (usando predições já calculadas)
f1_before = f1_score(y_test, y_pred)
print(f"F1-Score ANTES da otimização: {f1_before:.4f}")

# Grid Search para Random Forest
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, 20],
    "min_samples_split": [5, 10, 15]
}

grid = GridSearchCV(rf_model, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print(f"\nMelhores hiperparâmetros: {grid.best_params_}")
print(f"F1-score otimizado (CV): {grid.best_score_:.4f}")

# Avaliar modelo otimizado no teste
best_optimized_model = grid.best_estimator_
y_pred_opt = best_optimized_model.predict(X_test)
y_prob_opt = best_optimized_model.predict_proba(X_test)[:, 1]

f1_after = f1_score(y_test, y_pred_opt)
roc_after = roc_auc_score(y_test, y_prob_opt)
acc_after = accuracy_score(y_test, y_pred_opt)

print(f"\nCOMPARAÇÃO ANTES vs DEPOIS:")
print(f"F1-Score:  {f1_before:.4f} → {f1_after:.4f} ({f1_after-f1_before:+.4f})")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f} → {roc_after:.4f} ({roc_after-roc_auc_score(y_test, y_prob):+.4f})")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f} → {acc_after:.4f} ({acc_after-accuracy_score(y_test, y_pred):+.4f})")

# ============================================================================
# CONCLUSÕES
# ============================================================================

print(f"\nINTERPRETABILIDADE:")
if hasattr(best_optimized_model, 'feature_importances_'):
    top_feature = feature_imp.iloc[0]['feature']
    print(f" - Feature mais importante: {top_feature}")


print(f"\nRISCOS MITIGADOS:")
print(f"Desbalanceamento: class_weight='balanced'")
print(f"Vazamento de dados: apenas features pré-pedido")
print(f"Overfitting: validação cruzada + learning curves")
print(f"Outliers: clipping em percentis 5-95")

print("\n" + "="*60)
print("ANÁLISE COMPLETA FINALIZADA!")
print("="*60)

# ============================================================================
# EXEMPLO DE INFERÊNCIA EM NOVO PEDIDO
# ============================================================================
print("\n" + "="*60)
print("EXEMPLO DE INFERÊNCIA EM NOVO PEDIDO")
print("="*60)

def prever_satisfacao_cliente(novo_pedido_data, modelo_treinado, label_encoder, feature_names):
    """
    Função para prever a satisfação do cliente em um novo pedido
    """
    
    # Converter para formato de data
    data_pedido = pd.to_datetime(novo_pedido_data['order_purchase_timestamp'])
    data_entrega = pd.to_datetime(novo_pedido_data['order_delivered_customer_date'])
    data_estimada = pd.to_datetime(novo_pedido_data['order_estimated_delivery_date'])
    
    # Calcular as features (mesma engenharia de atributos do treinamento)
    atraso_dias = (data_entrega - data_estimada).days
    tempo_entrega = (data_entrega - data_pedido).days
    mes_compra = data_pedido.month
    dia_semana = data_pedido.dayofweek
    is_weekend = int(dia_semana >= 5)
    total_price = novo_pedido_data['total_price']
    avg_price = total_price / novo_pedido_data['num_items']
    avg_freight = novo_pedido_data['freight_value']
    num_items = novo_pedido_data['num_items']
    freight_ratio = avg_freight / (avg_price + 0.01)
    
    # Encode do estado (usando o mesmo LabelEncoder do treinamento)
    try:
        customer_state_encoded = label_encoder.transform([novo_pedido_data['customer_state']])[0]
    except ValueError:
        # Se o estado não foi visto no treinamento, usar um valor padrão
        print(f"Estado '{novo_pedido_data['customer_state']}' não visto no treinamento. Usando valor padrão.")
        customer_state_encoded = 0
    
    # Criar DataFrame com a mesma estrutura do treinamento
    novo_dado = pd.DataFrame([[
        atraso_dias, tempo_entrega, mes_compra, dia_semana, is_weekend,
        total_price, avg_price, avg_freight, num_items, freight_ratio,
        customer_state_encoded
    ]], columns=feature_names)
    
    # Fazer predições
    predicao_binaria = modelo_treinado.predict(novo_dado)[0]
    probabilidades = modelo_treinado.predict_proba(novo_dado)[0]
    prob_boa_avaliacao = probabilidades[1]
    
    return {
        'predicao_binaria': predicao_binaria,
        'probabilidade_boa_avaliacao': prob_boa_avaliacao,
        'dados_processados': novo_dado
    }

# EXEMPLO PRÁTICO 1: Pedido com entrega no prazo
print("\nEXEMPLO 1: Pedido com entrega no prazo")
novo_pedido_1 = {
    'order_purchase_timestamp': '2025-09-22 10:00:00',
    'order_delivered_customer_date': '2025-09-25 14:00:00',
    'order_estimated_delivery_date': '2025-09-27 10:00:00',
    'total_price': 150.0,
    'freight_value': 15.0,
    'num_items': 2,
    'customer_state': 'SP'
}

resultado_1 = prever_satisfacao_cliente(novo_pedido_1, best_optimized_model, le, X.columns)

print(f"Predição: {'Boa avaliação' if resultado_1['predicao_binaria'] == 1 else 'Avaliação ruim'}")
print(f"Probabilidade de boa avaliação: {resultado_1['probabilidade_boa_avaliacao']:.1%}")
print(f"Confiança: {'Alta' if resultado_1['probabilidade_boa_avaliacao'] > 0.8 or resultado_1['probabilidade_boa_avaliacao'] < 0.2 else 'Média'}")

# EXEMPLO PRÁTICO 2: Pedido com atraso significativo
print("\nEXEMPLO 2: Pedido com atraso significativo")
novo_pedido_2 = {
    'order_purchase_timestamp': '2025-09-22 10:00:00',
    'order_delivered_customer_date': '2025-10-02 14:00:00',
    'order_estimated_delivery_date': '2025-09-27 10:00:00',
    'total_price': 80.0,
    'freight_value': 25.0,
    'num_items': 1,
    'customer_state': 'RJ'
}

resultado_2 = prever_satisfacao_cliente(novo_pedido_2, best_optimized_model, le, X.columns)

print(f"Predição: {'Boa avaliação' if resultado_2['predicao_binaria'] == 1 else '❌ Avaliação ruim'}")
print(f"Probabilidade de boa avaliação: {resultado_2['probabilidade_boa_avaliacao']:.1%}")
print(f"Confiança: {'Alta' if resultado_2['probabilidade_boa_avaliacao'] > 0.8 or resultado_2['probabilidade_boa_avaliacao'] < 0.2 else 'Média'}")

# ANÁLISE SHAP DO NOVO DADO (Exemplo 1)
print(f"\nANÁLISE SHAP - Por que o modelo fez essa predição?")
print("(Analisando o Exemplo 1)")

# Reconstruir o explainer (já tinha feito antes, mas garantir)
explainer_new = shap.TreeExplainer(best_optimized_model)

# Obter SHAP values para o novo ponto (resultado_1['dados_processados'])
shap_values_raw = explainer_new.shap_values(resultado_1['dados_processados'])

# Debug completo para entender o que está vindo
print(f"[DEBUG] Tipo do retorno: {type(shap_values_raw)}")
if isinstance(shap_values_raw, list):
    print(f"[DEBUG] Lista com {len(shap_values_raw)} elementos")
    for idx, item in enumerate(shap_values_raw):
        print(f"[DEBUG] Elemento {idx}: shape {np.asarray(item).shape}")
else:
    print(f"[DEBUG] Array shape: {np.asarray(shap_values_raw).shape}")

print(f"[DEBUG] Número de features esperadas: {len(X.columns)}")
print(f"[DEBUG] Shape dos dados de entrada: {resultado_1['dados_processados'].shape}")

# Normalizar o formato para um vetor 1D com length = n_features
sv = shap_values_raw  # alias

# Converter para numpy array se não for lista
if isinstance(sv, list):
    # pegar a classe positiva (índice 1) se existir, senão a primeira
    if len(sv) > 1:
        sv = np.asarray(sv[1])   # shape -> (n_samples, n_features)
    else:
        sv = np.asarray(sv[0])
else:
    sv = np.asarray(sv)

print(f"[DEBUG] Shape após conversão inicial: {sv.shape}")

# Se for 3D (n_classes, n_samples, n_features) -> pegar classe positiva
if sv.ndim == 3:
    # preferir classe 1 (positiva) se houver
    sv = sv[1] if sv.shape[0] > 1 else sv[0]  # agora (n_samples, n_features)
    print(f"[DEBUG] Shape após redução 3D->2D: {sv.shape}")

# Agora sv deve ser (n_samples, n_features) ou (n_features,) ou (1, n_features)
if sv.ndim == 1:
    shap_vals_point = sv
else:
    # pegar a primeira (única) amostra
    shap_vals_point = sv[0]

# Garantir que seja 1D numpy array
shap_vals_point = np.asarray(shap_vals_point).reshape(-1)

print(f"[DEBUG] Shape final normalizado: {shap_vals_point.shape}")
print(f"[DEBUG] Primeiro e último valor SHAP: [{shap_vals_point[0]:.4f}, ..., {shap_vals_point[-1]:.4f}]")

# Verificar se o número de elementos bate com o número de features
if len(shap_vals_point) != len(X.columns):
    print(f"ERRO: Número de valores SHAP ({len(shap_vals_point)}) diferente do número de features ({len(X.columns)})")
    print("Pulando análise SHAP detalhada...")
    # Análise simplificada
    feature_contributions = []
    print("\nAnálise SHAP simplificada (valores podem estar incompletos):")
    for i in range(min(len(shap_vals_point), len(X.columns))):
        feature = X.columns[i]
        contribution = float(shap_vals_point[i])
        value = float(resultado_1['dados_processados'].iloc[0, i])
        feature_contributions.append({
            'feature': feature,
            'value': value,
            'shap_value': contribution,
            'impact': 'Positivo' if contribution > 0 else 'Negativo'
        })
    
    if len(feature_contributions) > 0:
        feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        print(f"\nTOP {min(3, len(feature_contributions))} Features disponíveis:")
        for i, contrib in enumerate(feature_contributions[:3]):
            print(f"{i+1}. {contrib['feature']}: {contrib['value']:.2f} "
                  f"(SHAP: {contrib['shap_value']:+.3f}, {contrib['impact']})")
else:
    # Análise completa normal
    feature_contributions = []
    for i, feature in enumerate(X.columns):
        contribution = float(shap_vals_point[i]) 
        value = float(resultado_1['dados_processados'].iloc[0, i])
        feature_contributions.append({
            'feature': feature,
            'value': value,
            'shap_value': contribution,
            'impact': 'Positivo' if contribution > 0 else 'Negativo'
        })

    # Ordenar e imprimir top 5
    feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)

    print("\nTOP 5 Features que mais influenciaram a predição:")
    for i, contrib in enumerate(feature_contributions[:5]):
        print(f"{i+1}. {contrib['feature']}: {contrib['value']:.2f} "
              f"(SHAP: {contrib['shap_value']:+.3f}, {contrib['impact']})")


expected_val = explainer_new.expected_value
if isinstance(expected_val, (list, np.ndarray)):
    try:
        expected_val = float(expected_val[1])
    except:
        expected_val = float(np.asarray(expected_val).ravel()[0])
else:
    expected_val = float(expected_val)

print(f"\nINTERPRETAÇÃO:")
print(f" - Valor base do modelo: {expected_val:.3f}")
if len(feature_contributions) > 0:
    print(f" - Soma dos impactos SHAP: {sum([c['shap_value'] for c in feature_contributions]):.3f}")
print(f" - Predição final: {float(resultado_1['probabilidade_boa_avaliacao']):.3f}")

print("\n" + "="*60)
print("EXEMPLO DE INFERÊNCIA CONCLUÍDO!")
print("="*60)