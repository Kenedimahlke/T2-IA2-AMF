# %% ---------------------------
# Imports
# ---------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, classification_report,
                             roc_auc_score, roc_curve, precision_recall_curve, auc, f1_score, average_precision_score)
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer

# Modelos
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
sns.set(style='whitegrid')

# %% ---------------------------
# Carregar Dataset
# ---------------------------
orders = pd.read_csv('datasets/olist_orders_dataset.csv')
reviews = pd.read_csv('datasets/olist_order_reviews_dataset.csv')
items = pd.read_csv('datasets/olist_order_items_dataset.csv')
customers = pd.read_csv('datasets/olist_customers_dataset.csv')

print("Datasets carregados com sucesso!")
print(f"Orders: {orders.shape}")
print(f"Reviews: {reviews.shape}")
print(f"Items: {items.shape}")
print(f"Customers: {customers.shape}")

# %% ---------------------------
# Pré-processamento e Engenharia de Features
# ---------------------------
# Remover linhas sem datas essenciais
orders = orders.dropna(subset=['order_delivered_customer_date', 'order_estimated_delivery_date'])

# Preenchimentos
orders['order_approved_at'] = orders['order_approved_at'].fillna(orders['order_purchase_timestamp'])
orders['order_delivered_carrier_date'] = orders['order_delivered_carrier_date'].fillna(orders['order_estimated_delivery_date'])

# Datas e features derivadas
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_approved_at'] = pd.to_datetime(orders['order_approved_at'])

orders['atraso'] = (orders['order_delivered_customer_date'] > orders['order_estimated_delivery_date']).astype(int)
orders['atraso_entrega'] = (orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']).dt.days
orders['tempo_aprovacao'] = (orders['order_approved_at'] - orders['order_purchase_timestamp']).dt.total_seconds() / 3600
orders['tempo_entrega'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days
orders['mes_compra'] = orders['order_purchase_timestamp'].dt.month
orders['dia_semana_compra'] = orders['order_purchase_timestamp'].dt.dayofweek

# Tratar freight_value outliers (IQR sobre items)
Q1 = items['freight_value'].quantile(0.25)
Q3 = items['freight_value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
items = items[(items['freight_value'] >= lower_bound) & (items['freight_value'] <= upper_bound)]

# Garantir price
if 'price' not in items.columns:
    preco_medio_produto = items.groupby('product_id')['price'].mean().reset_index()
    preco_medio_produto.rename(columns={'price': 'preco_medio'}, inplace=True)
    items = items.merge(preco_medio_produto, on='product_id', how='left')
    items.rename(columns={'preco_medio': 'price'}, inplace=True)

items['frete_relativo'] = items['freight_value'] / items['price']

# num_itens por pedido
num_itens = items.groupby('order_id').size().reset_index(name='num_itens')
items = items.merge(num_itens, on='order_id', how='left')

# histórico do seller
seller_reviews = reviews.merge(items[['order_id', 'seller_id']], on='order_id', how='left')
historico_seller = seller_reviews.groupby('seller_id')['review_score'].mean().reset_index(name='media_reviews_seller')
items = items.merge(historico_seller, on='seller_id', how='left')

# Merge orders + customers + reviews + items-aggregados (keep only one row per order)
orders = orders.merge(customers[['customer_id', 'customer_state']], on='customer_id', how='left')
reviews['review_good'] = (reviews['review_score'] >= 4).astype(int)

final = orders.merge(reviews[['order_id', 'review_good']], on='order_id', how='left')
agg_items = items.groupby('order_id').agg({
    'frete_relativo': 'mean',
    'num_itens': 'mean',
    'media_reviews_seller': 'mean'
}).reset_index()
final = final.merge(agg_items, on='order_id', how='left')

# Remover linhas sem target
final = final.dropna(subset=['review_good'])

# Selecionar features
features = ['atraso', 'atraso_entrega', 'tempo_aprovacao', 'tempo_entrega',
            'mes_compra', 'dia_semana_compra', 'frete_relativo', 'num_itens', 'media_reviews_seller', 'order_status']

X = final[features].copy()
y = final['review_good'].astype(int)

print(f"Dataset final: {X.shape} features, {y.value_counts().to_dict()} distribuição classes")

# %% ---------------------------
# Preparação dos Transformadores
# ---------------------------
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns

num_transformer = SkPipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = SkPipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ]
)

print(f"Features numéricas: {list(num_cols)}")
print(f"Features categóricas: {list(cat_cols)}")

# %% ---------------------------
# Divisão dos dados
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Baseline: prever a classe majoritária do treino
major_class = y_train.mode()[0]
baseline_pred = np.full_like(y_test, fill_value=major_class)
baseline_f1 = f1_score(y_test, baseline_pred)
print(f"Baseline (major class={major_class}): F1 = {baseline_f1:.4f}")

# %% ---------------------------
# Comparação de Modelos
# ---------------------------
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42)
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("="*60)
print("TREINAMENTO E COMPARAÇÃO DE ALGORITMOS")
print("="*60)

for name, model in models.items():
    print(f"Treinando {name}...")
    model_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('resampler', SMOTETomek(random_state=42)),
        ('clf', model)
    ])
    cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'pipeline': model_pipeline
    }
    print(f"{name}: F1 = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Melhor modelo
best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
best_pipeline = results[best_model_name]['pipeline']
best_pipeline.fit(X_train, y_train)

print(f"\nMelhor modelo: {best_model_name} com F1 médio: {results[best_model_name]['cv_mean']:.4f}")

# %% ---------------------------
# Avaliação no Teste
# ---------------------------
y_pred = best_pipeline.predict(X_test)
y_probs = best_pipeline.predict_proba(X_test)[:, 1]

print("="*60)
print("AVALIAÇÃO NO CONJUNTO DE TESTE")
print("="*60)

f1_test = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probs)
pr_auc = average_precision_score(y_test, y_probs)

print("Classification Report (Teste):")
print(classification_report(y_test, y_pred))
print(f"F1-Score: {f1_test:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC")
plt.legend()
plt.savefig('roc_curve.png')
plt.close()

# Curva Precisão-Recall
prec, rec, _ = precision_recall_curve(y_test, y_probs)

plt.figure(figsize=(6, 5))
plt.plot(rec, prec, label=f"AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precisão-Recall")
plt.legend()
plt.savefig('precision_recall_curve.png')
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, normalize='true')
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues')
plt.title('Confusion Matrix (normalized)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_normalized.png')
plt.close()

# %% ---------------------------
# Importância de Variáveis
# ---------------------------
print("="*60)
print("ANÁLISE DE IMPORTÂNCIA DAS VARIÁVEIS")
print("="*60)

# Ajustar preprocessor para obter nomes das features
preprocessor.fit(X_train)

# Permutation Importance
perm_importance = permutation_importance(best_pipeline, X_test, y_test, n_repeats=10, random_state=42, scoring='f1', n_jobs=-1)

try:
    feature_names = preprocessor.get_feature_names_out()
except:
    # Fallback caso get_feature_names_out não funcione
    feature_names = [f"feature_{i}" for i in range(len(perm_importance.importances_mean))]

importances = pd.DataFrame({
    'feature': feature_names[:len(perm_importance.importances_mean)],
    'importance': perm_importance.importances_mean
}).sort_values(by='importance', ascending=False)

print("Top 10 Features - Permutation Importance:")
print(importances.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(data=importances.head(15), x="importance", y="feature", palette="viridis")
plt.title("Top 15 - Permutation Importance")
plt.tight_layout()
plt.savefig('permutation_importance.png')
plt.close()

# %% ---------------------------
# SHAP Values
# ---------------------------
print("="*60)
print("ANÁLISE SHAP")
print("="*60)

explainer = shap.TreeExplainer(best_pipeline.named_steps['clf'])
X_sample = X_test.sample(100, random_state=42)
X_sample_trans = preprocessor.transform(X_sample)

shap_values = explainer.shap_values(X_sample_trans)

print(f"Tipo de shap_values: {type(shap_values)}")
if isinstance(shap_values, list):
    print(f"Número de classes: {len(shap_values)}")
    shap_values_to_plot = shap_values[1]  # Classe positiva
else:
    shap_values_to_plot = shap_values

# SHAP Summary Plot
shap.summary_plot(shap_values_to_plot, X_sample_trans, feature_names=feature_names[:X_sample_trans.shape[1]], show=False)
plt.title('SHAP Values - Importância e Impacto das Features')
plt.tight_layout()
plt.savefig('shap_summary.png')
plt.close()

# SHAP Waterfall para 1 exemplo
try:
    shap.plots.waterfall(shap.Explanation(
        values=shap_values_to_plot[0],
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        data=X_sample_trans[0],
        feature_names=feature_names[:X_sample_trans.shape[1]]
    ))
    plt.savefig('shap_waterfall.png')
    plt.close()
except Exception as e:
    print(f"Erro ao gerar waterfall plot: {e}")

print("Gráficos SHAP salvos com sucesso!")

# %% ---------------------------
# Learning Curve
# ---------------------------
print("="*60)
print("LEARNING CURVES")
print("="*60)

train_sizes, train_scores, test_scores = learning_curve(
    best_pipeline, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='f1', n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Treino')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(train_sizes, test_mean, 'o-', color='red', label='Validação')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')

plt.xlabel("Tamanho do Conjunto de Treino")
plt.ylabel("F1-score")
plt.title("Learning Curves - Detecção de Overfitting")
plt.legend()
plt.grid(True)
plt.savefig('learning_curves.png')
plt.close()

# Análise de overfitting
gap = train_mean[-1] - test_mean[-1]
print(f"Gap final treino-validação: {gap:.4f}")
if gap > 0.05:
    print("⚠️ Possível overfitting detectado!")
else:
    print("✅ Modelo bem generalizado")

# %% ---------------------------
# Comparação de Modelos (Tabela Final)
# ---------------------------
print("="*60)
print("RESUMO FINAL DOS RESULTADOS")
print("="*60)

results_df = pd.DataFrame({
    'Modelo': list(results.keys()) + ['Baseline'],
    'F1 CV Médio': [results[m]['cv_mean'] for m in results] + [baseline_f1],
    'F1 CV Std': [results[m]['cv_std'] for m in results] + [0.0]
}).sort_values(by="F1 CV Médio", ascending=False)

print("Resumo dos Modelos Testados:")
print(results_df.to_string(index=False))

print(f"\n🏆 MELHOR MODELO: {best_model_name}")
print(f"📊 F1-Score no teste: {f1_test:.4f}")
print(f"📈 ROC-AUC: {roc_auc:.4f}")
print(f"🎯 Melhoria sobre baseline: {((f1_test/baseline_f1 - 1) * 100):.1f}%")

print("\n📁 Arquivos gerados:")
print("- confusion_matrix_normalized.png")
print("- roc_curve.png")
print("- precision_recall_curve.png")
print("- permutation_importance.png")
print("- shap_summary.png")
print("- learning_curves.png")

print("\n" + "="*60)
print("ANÁLISE COMPLETA FINALIZADA!")
print("="*60)
