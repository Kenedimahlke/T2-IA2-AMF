# ---------------------------
# Imports
# ---------------------------
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # CORREÇÃO #1: Backend para evitar tkinter
import matplotlib.pyplot as plt
import warnings
import joblib
from math import radians, sin, cos, sqrt, atan2

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
from sklearn.base import clone  # CORREÇÃO #2: Adicionar clone
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_recall_curve, f1_score, average_precision_score,
                             accuracy_score, precision_score, recall_score)
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer

# Modelos
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# MELHORIA: Adicionar LightGBM
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
    print("[OK] LightGBM disponível")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[WARN] LightGBM não disponível, usando Random Forest")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] SHAP não disponível")

warnings.filterwarnings("ignore")

print("INICIANDO ANÁLISE RANDOM FOREST MELHORADO...")
print("="*60)

# CORREÇÃO: Configuração robusta do diretório de outputs
output_dir = "outputs"
try:
    os.makedirs(output_dir, exist_ok=True)
    # Verificar se o diretório é escribível
    test_file = os.path.join(output_dir, "test_write.tmp")
    with open(test_file, 'w') as f:
        f.write("test")
    os.remove(test_file)
    print(f"[OK] Diretório '{output_dir}' criado/verificado e acessível")
except PermissionError:
    print(f"[ERROR] Sem permissão para escrever em '{output_dir}'. Usando diretório atual.")
    output_dir = "."
except Exception as e:
    print(f"[ERROR] Erro ao configurar diretório de outputs: {e}. Usando diretório atual.")
    output_dir = "."

# ---------------------------
# Carregar Dataset
# ---------------------------
print("Carregando datasets...")

# CORREÇÃO: Validação robusta dos dados de entrada
def safe_load_dataset(filepath, dataset_name):
    """Carregar dataset com validação robusta"""
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            raise ValueError(f"Dataset {dataset_name} está vazio")
        print(f"   [OK] {dataset_name}: {df.shape} - {df.memory_usage(deep=True).sum() / 1024**2:.1f}MB")
        return df
    except FileNotFoundError:
        print(f"   [ERROR] {dataset_name}: Arquivo não encontrado em {filepath}")
        raise
    except Exception as e:
        print(f"   [ERROR] {dataset_name}: Erro ao carregar - {e}")
        raise

# Carregar datasets principais (obrigatórios)
try:
    orders = safe_load_dataset('datasets/olist_orders_dataset.csv', 'Orders')
    reviews = safe_load_dataset('datasets/olist_order_reviews_dataset.csv', 'Reviews')
    items = safe_load_dataset('datasets/olist_order_items_dataset.csv', 'Items')
    customers = safe_load_dataset('datasets/olist_customers_dataset.csv', 'Customers')
    print("[OK] Datasets principais carregados com sucesso!")
except Exception as e:
    print(f"[ERROR] Erro crítico ao carregar datasets principais: {e}")
    print("[STOP] Não é possível continuar sem os datasets principais")
    raise

# CORREÇÃO: Validação robusta para datasets adicionais
try:
    products = safe_load_dataset('datasets/olist_products_dataset.csv', 'Products')
    geolocation = safe_load_dataset('datasets/olist_geolocation_dataset.csv', 'Geolocation')
    sellers = safe_load_dataset('datasets/olist_sellers_dataset.csv', 'Sellers')
    
    # Validar integridade básica dos datasets adicionais
    required_columns = {
        'products': ['product_id', 'product_category_name'],
        'geolocation': ['geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng'],
        'sellers': ['seller_id', 'seller_zip_code_prefix']
    }
    
    datasets_check = {
        'products': products,
        'geolocation': geolocation, 
        'sellers': sellers
    }
    
    for name, df in datasets_check.items():
        missing_cols = set(required_columns[name]) - set(df.columns)
        if missing_cols:
            print(f"   [WARN] {name}: Colunas obrigatórias ausentes: {missing_cols}")
            ADVANCED_FEATURES = False
            break
    else:
        print("[OK] Datasets adicionais validados com sucesso!")
        ADVANCED_FEATURES = True
        
except FileNotFoundError as e:
    print(f"[WARN] Alguns datasets adicionais não encontrados: {e}")
    print("Continuando com features básicas...")
    ADVANCED_FEATURES = False
except Exception as e:
    print(f"[WARN] Erro ao validar datasets adicionais: {e}")
    print("Continuando com features básicas...")
    ADVANCED_FEATURES = False

# ---------------------------
# Pré-processamento e Engenharia de Features
# ---------------------------
print("\nPRÉ-PROCESSAMENTO E FEATURE ENGINEERING...")

# Remover linhas sem datas essenciais
orders = orders.dropna(subset=['order_delivered_customer_date', 'order_estimated_delivery_date'])

# Preenchimentos
orders['order_approved_at'] = orders['order_approved_at'].fillna(orders['order_purchase_timestamp'])

# Converter colunas de data
date_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_customer_date', 'order_estimated_delivery_date']
for col in date_cols:
    orders[col] = pd.to_datetime(orders[col])

# Calcular features temporais
orders['atraso'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days
orders['atraso_entrega'] = (orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']).dt.days
orders['tempo_aprovacao'] = (orders['order_approved_at'] - orders['order_purchase_timestamp']).dt.total_seconds() / 3600  # horas
orders['tempo_entrega'] = (orders['order_delivered_customer_date'] - orders['order_approved_at']).dt.days

# Features temporais adicionais
orders['mes_compra'] = orders['order_purchase_timestamp'].dt.month
orders['dia_semana_compra'] = orders['order_purchase_timestamp'].dt.dayofweek
orders['hora_compra'] = orders['order_purchase_timestamp'].dt.hour
orders['is_weekend'] = (orders['dia_semana_compra'] >= 5).astype(int)

# Status do pedido
orders['order_finalizado'] = (orders['order_status'] == 'delivered').astype(int)

# Merge orders com reviews
reviews['review_good'] = (reviews['review_score'] >= 4).astype(int)
final = orders.merge(reviews[['order_id', 'review_good']], on='order_id', how='inner')

# Tratamento de outliers nos dados de frete
items['freight_value'] = items['freight_value'].fillna(0)
q95 = items['freight_value'].quantile(0.95)
q5 = items['freight_value'].quantile(0.05)
items['freight_value'] = items['freight_value'].clip(lower=q5, upper=q95)
print(f"Freight outliers tratados por capping: [{q5:.2f}, {q95:.2f}]")

# Feature engineering avançada dos items
items['frete_relativo'] = items['freight_value'] / (items['price'] + 0.01)  # Evitar divisão por zero

# MELHORIA: Feature de categoria do produto
if ADVANCED_FEATURES:
    print("Aplicando feature engineering avançada...")
    items = items.merge(products[['product_id', 'product_category_name']], on='product_id', how='left')
    items['product_category_name'] = items['product_category_name'].fillna('outros')

# Histórico de reviews por seller (temporal) - CORREÇÃO: Versão vetorizada
items_with_reviews = items.merge(final[['order_id', 'order_purchase_timestamp', 'review_good']], on='order_id', how='left')

print("Calculando histórico de seller (versão vetorizada)...")
# CORREÇÃO: Algoritmo vetorizado O(N log N) em vez de O(N²) - Fix para pandas compatibility
items_with_reviews = items_with_reviews.sort_values(['seller_id', 'order_purchase_timestamp']).reset_index(drop=True)

# Calcular média acumulada excluindo a própria ordem usando transform (mais eficiente)
def calc_rolling_mean_transform(group):
    """Calcula média acumulada excluindo o próprio registro usando transform"""
    shifted = group.shift()
    return shifted.expanding().mean()

# Usar transform em vez de apply para evitar warnings e melhor performance
items_with_reviews['media_reviews_seller'] = (
    items_with_reviews.groupby('seller_id')['review_good']
    .transform(calc_rolling_mean_transform)
)

# Estatísticas do histórico calculado
valid_history = items_with_reviews['media_reviews_seller'].notna().sum()
total_sellers = items_with_reviews['seller_id'].nunique()
print(f"[OK] Histórico seller calculado (vetorizado): {total_sellers} sellers, {valid_history} reviews com histórico")

# Agregar dados por pedido - CORREÇÃO: Agregação mais robusta
if ADVANCED_FEATURES:
    def safe_mode(x):
        """Função segura para calcular moda, lidando com valores vazios"""
        try:
            mode_result = x.mode()
            if not mode_result.empty:
                return mode_result.iloc[0]
            else:
                return 'outros'
        except:
            return 'outros'
    
    agg_items = items_with_reviews.groupby('order_id').agg(
        frete_relativo=('frete_relativo', 'mean'),
        num_itens=('order_item_id', 'count'),
        media_reviews_seller=('media_reviews_seller', 'mean'),
        product_category_name=('product_category_name', safe_mode)
    ).reset_index()
else:
    agg_items = items_with_reviews.groupby('order_id').agg(
        frete_relativo=('frete_relativo', 'mean'),
        num_itens=('order_item_id', 'count'),
        media_reviews_seller=('media_reviews_seller', 'mean')
    ).reset_index()

final = final.merge(agg_items, on='order_id', how='left')

# MELHORIA: Feature de distância geográfica
if ADVANCED_FEATURES:
    print("Calculando distância geográfica vendedor-cliente...")
    
    # Função para calcular distância haversine
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Raio da Terra em km
        try:
            if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
                return np.nan
            dlat = radians(lat2 - lat1)
            dlon = radians(lon2 - lon1)
            a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return R * c
        except (ValueError, TypeError):
            return np.nan
    
    # CORREÇÃO: Calcular distância geográfica com merge mais robusto
    print("Calculando distância geográfica vendedor-cliente...")
    
    # Preparar dados de geolocalização
    geo_agg = geolocation.groupby('geolocation_zip_code_prefix').first().reset_index()
    
    # Merge com dados de cliente e seller
    final = final.merge(customers[['customer_id', 'customer_zip_code_prefix', 'customer_state']], on='customer_id', how='left')
    
    # Pegar seller_id através dos items
    order_seller = items.groupby('order_id')['seller_id'].first().reset_index()
    final = final.merge(order_seller, on='order_id', how='left')
    final = final.merge(sellers[['seller_id', 'seller_zip_code_prefix']], on='seller_id', how='left')
    
    # Merge coordenadas do cliente
    customer_geo = geo_agg[['geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']].copy()
    customer_geo = customer_geo.rename(columns={
        'geolocation_zip_code_prefix': 'customer_zip_code_prefix',
        'geolocation_lat': 'customer_lat', 
        'geolocation_lng': 'customer_lng'
    })
    final = final.merge(customer_geo, on='customer_zip_code_prefix', how='left')
    
    # Merge coordenadas do seller
    seller_geo = geo_agg[['geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']].copy()
    seller_geo = seller_geo.rename(columns={
        'geolocation_zip_code_prefix': 'seller_zip_code_prefix',
        'geolocation_lat': 'seller_lat', 
        'geolocation_lng': 'seller_lng'
    })
    final = final.merge(seller_geo, on='seller_zip_code_prefix', how='left')
    
    # Verificar se as colunas existem antes de calcular distância
    required_cols = ['seller_lat', 'seller_lng', 'customer_lat', 'customer_lng']
    missing_cols = [col for col in required_cols if col not in final.columns]
    
    if missing_cols:
        print(f"[WARN] Colunas ausentes para cálculo de distância: {missing_cols}")
        print("[WARN] Pulando cálculo de distância geográfica")
        final['distancia_vendedor_cliente_km'] = np.nan
    else:
        # Calcular distância apenas para linhas com coordenadas válidas
        mask = (final[required_cols].notna().all(axis=1))
        final['distancia_vendedor_cliente_km'] = np.nan
        
        if mask.sum() > 0:
            final.loc[mask, 'distancia_vendedor_cliente_km'] = final.loc[mask].apply(
                lambda row: haversine_distance(row['seller_lat'], row['seller_lng'], 
                                             row['customer_lat'], row['customer_lng']), axis=1
            )
            print(f"[OK] Distância calculada para {final['distancia_vendedor_cliente_km'].notna().sum()} pedidos")
        else:
            print("[WARN] Nenhuma coordenada válida encontrada para cálculo de distância")

# Remover linhas sem target
final = final.dropna(subset=['review_good'])

# ---------------------------
# Selecionar features
# ---------------------------
print("\nSELEÇÃO DE FEATURES...")

# Features básicas
basic_features = [
    'atraso', 'atraso_entrega', 'tempo_aprovacao', 'tempo_entrega',
    'mes_compra', 'dia_semana_compra', 'hora_compra', 'is_weekend',
    'order_finalizado', 'frete_relativo', 'num_itens', 'media_reviews_seller'
]

# MELHORIA: Adicionar features avançadas se disponíveis
if ADVANCED_FEATURES:
    advanced_features = ['customer_state', 'product_category_name']
    if 'distancia_vendedor_cliente_km' in final.columns:
        advanced_features.append('distancia_vendedor_cliente_km')
    features = basic_features + advanced_features
    print(f"[OK] Usando {len(advanced_features)} features avançadas")
else:
    features = basic_features
    print("📝 Usando apenas features básicas")

X = final[features].copy()
y = final['review_good'].astype(int)

# Tratar NaNs
if ADVANCED_FEATURES:
    if 'customer_state' in X.columns:
        X['customer_state'] = X['customer_state'].fillna('Desconhecido')
    if 'product_category_name' in X.columns:
        X['product_category_name'] = X['product_category_name'].fillna('outros')
    if 'distancia_vendedor_cliente_km' in X.columns:
        X['distancia_vendedor_cliente_km'] = X['distancia_vendedor_cliente_km'].fillna(X['distancia_vendedor_cliente_km'].median())

X['media_reviews_seller'] = X['media_reviews_seller'].fillna(0.5)  # Valor neutro

print(f"Dataset final: {X.shape} features, {y.value_counts().to_dict()} distribuição classes")

# Identificar features numéricas e categóricas
num_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
cat_cols = [col for col in X.columns if X[col].dtype == 'object']

print(f"Features numéricas: {num_cols}")
print(f"Features categóricas: {cat_cols}")

# ---------------------------
# Split dos dados
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ---------------------------
# Pipeline de preprocessamento
# ---------------------------
print("\nCRIANDO PIPELINE DE PREPROCESSAMENTO...")

# Transformers
num_transformer = SkPipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

if cat_cols:
    cat_transformer = SkPipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    preprocessor_base = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )
else:
    preprocessor_base = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols)
        ]
    )

print("[OK] Pipeline de preprocessamento criado")

# ---------------------------
# Baseline
# ---------------------------
print("\n📏 CALCULANDO BASELINE...")
major_class = y_train.value_counts().idxmax()
baseline_pred = np.full(y_test.shape, fill_value=major_class)
baseline_f1 = f1_score(y_test, baseline_pred)
print(f"Baseline (major class={major_class}): F1 = {baseline_f1:.4f}")

# ---------------------------
# Modelo e Cross-validation
# ---------------------------
print("="*60)
print("TREINAMENTO DO MODELO")
print("="*60)

# MELHORIA: Usar LightGBM se disponível
if LIGHTGBM_AVAILABLE:
    model_name = "LightGBM"
    model = LGBMClassifier(
        objective='binary',
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        colsample_bytree=0.8,
        verbosity=-1
    )
else:
    model_name = "Random Forest"
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

# Pipeline completo
final_pipeline = ImbPipeline([
    ('preprocessor', clone(preprocessor_base)),  # CORREÇÃO #2: Clone preprocessor
    ('resampler', SMOTETomek(random_state=42)),
    ('clf', model)
])

# Cross-validation - MELHORIA: Usar f1_macro para datasets desbalanceados
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(final_pipeline, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
print(f"{model_name}: F1 Macro CV = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Treinar modelo final
final_pipeline.fit(X_train, y_train)

print(f"\n[OK] Modelo treinado: {model_name}")

# ---------------------------
# Criar pipeline de inferência (SEM resampler) para avaliação
# ---------------------------
inference_pipeline = SkPipeline([
    ('preprocessor', clone(preprocessor_base)),  # CORREÇÃO #2: Clone preprocessor
    ('clf', final_pipeline.named_steps['clf'])
])
# Ajustar o preprocessor usando X_train
inference_pipeline.named_steps['preprocessor'].fit(X_train)

# ---------------------------
# Avaliação no conjunto de teste
# ---------------------------
print("="*60)
print("AVALIAÇÃO NO CONJUNTO DE TESTE")
print("="*60)

# Predições
y_pred = inference_pipeline.predict(X_test)
y_probs = inference_pipeline.predict_proba(X_test)[:, 1]

# Métricas principais
f1_test = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probs)
pr_auc = average_precision_score(y_test, y_probs)

print("Classification Report (Teste):")
print(classification_report(y_test, y_pred, digits=4))

# F1 específico para cada classe - usando pos_label para consistência
f1_class0 = f1_score(y_test, y_pred, pos_label=0)  # Classe 0 (ruins)
f1_class1 = f1_score(y_test, y_pred, pos_label=1)  # Classe 1 (boas)

print(f"\nMétricas Gerais:")
print(f"F1-Score Geral: {f1_test:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

print(f"\nMétricas por Classe:")
print(f"F1 (Classe 0 - Reviews Ruins): {f1_class0:.4f}")
print(f"F1 (Classe 1 - Reviews Boas): {f1_class1:.4f}")
print(f"Diferença F1 (Classe1 - Classe0): {f1_class1 - f1_class0:.4f}")

# ---------------------------
# Otimização de threshold usando conjunto de validação (SEM data snooping)
# ---------------------------
print("\n" + "="*50)
print("OTIMIZAÇÃO DE THRESHOLD (usando validação)")
print("="*50)

# Split de validação a partir do treino
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# Treinar modelo apenas no subset de treino
val_pipeline = ImbPipeline([
    ('preprocessor', clone(preprocessor_base)),  # CORREÇÃO #2: Clone preprocessor
    ('resampler', SMOTETomek(random_state=42)),
    ('clf', clone(model))  # CORREÇÃO #2: Clone model
])

val_pipeline.fit(X_train_split, y_train_split)

# Pipeline de inferência para validação  
inference_pipeline_val = SkPipeline([
    ('preprocessor', clone(preprocessor_base)),  # CORREÇÃO #2: Clone preprocessor
    ('clf', val_pipeline.named_steps['clf'])
])
inference_pipeline_val.named_steps['preprocessor'].fit(X_train_split)

# Predições no conjunto de validação
y_pred_val = inference_pipeline_val.predict(X_val_split)
if hasattr(inference_pipeline_val.named_steps['clf'], "predict_proba"):
    y_probs_val = inference_pipeline_val.predict_proba(X_val_split)[:, 1]
else:
    y_probs_val = np.zeros(len(y_val_split))

# CORREÇÃO #4: F1 baseline com pos_label explícito 
f1_val_baseline = f1_score(y_val_split, y_pred_val)
f1_class0_baseline = f1_score(y_val_split, y_pred_val, pos_label=0)  # Classe 0 (ruins)
f1_class1_baseline = f1_score(y_val_split, y_pred_val, pos_label=1)  # Classe 1 (boas)

# Otimizar threshold para maximizar F1 da classe 0
best_t = 0.5
best_f1_class0 = f1_class0_baseline
best_results = {
    'threshold': 0.5,
    'f1_class0': f1_class0_baseline,
    'f1_class1': f1_class1_baseline,
    'f1_weighted': f1_val_baseline
}

thresholds = np.linspace(0.1, 0.9, 81)

for t in thresholds:
    preds_t = (y_probs_val >= t).astype(int)
    
    # CORREÇÃO #4: F1 para cada classe com pos_label explícito
    f1_0 = f1_score(y_val_split, preds_t, pos_label=0)  # Classe 0 (ruins)
    f1_1 = f1_score(y_val_split, preds_t, pos_label=1)  # Classe 1 (boas)
    f1_weighted = f1_score(y_val_split, preds_t, average='weighted')
    
    if f1_0 > best_f1_class0:
        best_f1_class0 = f1_0
        best_t = t
        best_results = {
            'threshold': t,
            'f1_class0': f1_0,
            'f1_class1': f1_1,
            'f1_weighted': f1_weighted
        }

print(f"Threshold padrão (0.5) no conjunto de validação:")
print(f"  F1 Classe 0: {f1_class0_baseline:.4f}")
print(f"  F1 Classe 1: {f1_class1_baseline:.4f}")

print(f"\nMelhor threshold encontrado: {best_t:.3f}")
print(f"  F1 Classe 0: {best_results['f1_class0']:.4f} (+{best_results['f1_class0']-f1_class0_baseline:.4f})")
print(f"  F1 Classe 1: {best_results['f1_class1']:.4f} ({best_results['f1_class1']-f1_class1_baseline:+.4f})")
print(f"  F1 Weighted: {best_results['f1_weighted']:.4f} ({best_results['f1_weighted']-f1_val_baseline:+.4f})")

# Aplicar threshold otimizado ao conjunto de teste
y_pred_optimized = (y_probs >= best_t).astype(int)

# Recalcular todas as métricas no TESTE com o novo threshold
f1_test_optimized = f1_score(y_test, y_pred_optimized)
f1_class0_optimized = f1_score(y_test, y_pred_optimized, pos_label=0)  # Usando pos_label para clareza
f1_class1_optimized = f1_score(y_test, y_pred_optimized, pos_label=1)  # Usando pos_label para clareza

print(f"\nResultados no conjunto de TESTE com threshold otimizado (threshold={best_t:.3f}):")
print(f"  F1 Geral: {f1_test_optimized:.4f} (vs {f1_test:.4f} com threshold=0.5)")
print(f"  F1 Classe 0 (Ruins): {f1_class0_optimized:.4f} (vs {f1_class0:.4f} com threshold=0.5)")
print(f"  F1 Classe 1 (Boas): {f1_class1_optimized:.4f} (vs {f1_class1:.4f} com threshold=0.5)")

# ---------------------------
# Visualizações
# ---------------------------
print("\nGERANDO VISUALIZAÇÕES...")

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC")
plt.legend()
plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall")
plt.legend()
plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# Matriz de confusão
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred_optimized)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ruins', 'Boas'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues')
plt.title(f'Matriz de Confusão (threshold={best_t:.3f})')
plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Visualizações salvas em {output_dir}/")

# ---------------------------
# Análise de Importância das Variáveis
# ---------------------------
print("="*60)
print("ANÁLISE DE IMPORTÂNCIA DAS VARIÁVEIS - PERMUTATION")
print("="*60)

# CORREÇÃO: Feature names robustas usando API do preprocessor
try:
    # Usar API direta do preprocessor para garantir ordem e consistência
    preproc = inference_pipeline.named_steps['preprocessor']
    feature_names = preproc.get_feature_names_out()
    print(f"[OK] Feature names extraídos via API do preprocessor: {len(feature_names)} features")
    
except Exception as e:
    print(f"[WARN] API get_feature_names_out não disponível: {e}")
    
    # Fallback robusto: construir manualmente
    feature_names = []
    
    # Features numéricas
    num_names = [f"num__{col}" for col in num_cols]
    feature_names.extend(num_names)
    
    # Features categóricas (se existirem)
    if cat_cols:
        try:
            cat_transformer = preproc.named_transformers_['cat']
            ohe = cat_transformer.named_steps['onehot']
            cat_feature_names = ohe.get_feature_names_out(cat_cols)
            feature_names.extend(cat_feature_names)
        except Exception:
            # Fallback manual para versões antigas
            for col in cat_cols:
                unique_vals = X_train[col].astype(str).unique()[:10]  # Limitar a 10 valores
                cat_names = [f"{col}__{val}" for val in unique_vals]
                feature_names.extend(cat_names)
    
    print(f"[OK] Feature names construídos manualmente: {len(feature_names)} features")

# Ajustar tamanho se necessário
X_test_trans = inference_pipeline.named_steps['preprocessor'].transform(X_test)
if len(feature_names) != X_test_trans.shape[1]:
    print(f"Aviso: Ajustando feature_names de {len(feature_names)} para {X_test_trans.shape[1]} features")
    feature_names = feature_names[:X_test_trans.shape[1]]
    if len(feature_names) < X_test_trans.shape[1]:
        feature_names.extend([f"feature_{i}" for i in range(len(feature_names), X_test_trans.shape[1])])

# Permutation Importance
# CORREÇÃO: Otimizar permutation importance para performance
perm_importance = permutation_importance(inference_pipeline, X_test, y_test, n_repeats=5, random_state=42, scoring='f1', n_jobs=-1)  # Reduzido de 10 para 5

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=False)

print("Top 10 Features - Permutation Importance:")
print(importance_df.head(10).to_string(index=False))

# Plot importance
plt.figure(figsize=(10, 6))
top_features = importance_df.head(10)
plt.barh(range(len(top_features)), top_features['importance'][::-1])
plt.yticks(range(len(top_features)), top_features['feature'][::-1])
plt.xlabel('Importância (Permutation)')
plt.title('Top 10 Features - Permutation Importance')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'permutation_importance.png'), dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------
# SHAP Analysis
# ---------------------------
if SHAP_AVAILABLE:
    print("="*60)
    print("ANÁLISE SHAP")
    print("="*60)

    # CORREÇÃO #6: Subamostragem para SHAP (performance e memória) - MELHORIA: Reduzir ainda mais
    shap_sample_size = min(500, len(X_test))  # Reduzido de 1000 para 500
    shap_indices = np.random.choice(len(X_test), shap_sample_size, replace=False)
    X_test_shap = X_test.iloc[shap_indices]
    print(f"Amostras para SHAP: {len(X_test_shap)} (de {len(X_test)} totais)")

    # Transformar X_test_shap com o preprocessor (usado para SHAP)
    X_test_trans_shap = inference_pipeline.named_steps['preprocessor'].transform(X_test_shap)

    # Se o classificador for Tree-based, usar TreeExplainer; caso contrário usar shap.Explainer
    clf = inference_pipeline.named_steps['clf']

    # CORREÇÃO #7: Try/catch robusto para SHAP
    try:
        print("Calculando SHAP values...")
        if hasattr(clf, "predict_proba") and (hasattr(clf, "feature_importances_") or 'Forest' in str(type(clf).__name__) or 'LGBM' in str(type(clf).__name__)):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_test_trans_shap)
        else:
            explainer = shap.Explainer(clf.predict_proba, X_test_trans_shap)
            shap_values = explainer(X_test_trans_shap)
    except Exception as e:
        print(f"Erro na criação do explainer: {e}")
        try:
            # fallback geral
            explainer = shap.Explainer(clf, X_test_trans_shap)
            shap_values = explainer(X_test_trans_shap)
        except Exception as e2:
            print(f"SHAP falhou completamente: {e2}")
            print("Pulando análise SHAP...")
            shap_values = None

    # Normalizar formato para plot - CORREÇÃO #7: Try/except 
    if shap_values is not None:
        try:
            if hasattr(shap, 'Explanation') and isinstance(shap_values, shap.Explanation):
                # shap v0.40+ style com Explanation object
                vals = shap_values.values
                # se multi classe, tentar pegar a classe positiva (1)
                if vals.ndim == 3:
                    shap_to_plot = vals[:, :, 1]  # (n_samples, n_features) para classe 1
                else:
                    shap_to_plot = vals
            elif hasattr(shap_values, "values"):
                # shap_values tem atributo values
                vals = shap_values.values
                if vals.ndim == 3:
                    shap_to_plot = vals[:, :, 1]
                else:
                    shap_to_plot = vals
            else:
                # lista de arrays (formato clássico)
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    shap_to_plot = shap_values[1]  # classe positiva
                elif isinstance(shap_values, list):
                    shap_to_plot = shap_values[0]
                else:
                    shap_to_plot = shap_values
            print(f"SHAP values processados com sucesso. Shape: {shap_to_plot.shape}")
        except Exception as e:
            print(f"Erro ao processar SHAP values: {e}")
            # Fallback: assumir que é array numpy simples
            shap_to_plot = shap_values

        # Tentar obter feature_names compatíveis
        fnames = feature_names[:X_test_trans_shap.shape[1]]

        # CORREÇÃO #7: Summary plot protegido
        try:
            shap.summary_plot(shap_to_plot, X_test_trans_shap, feature_names=fnames, show=False)
            plt.title('SHAP Summary Plot - Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OK] SHAP summary plot salvo: {os.path.join(output_dir, 'shap_summary.png')}")
        except Exception as e:
            print(f"[WARN] Erro ao gerar summary plot: {e}")

        # MELHORIA: Waterfall plot para exemplo específico (primeira amostra de review ruim)
        try:
            # Encontrar primeira amostra de review ruim para explicação local
            y_test_shap = y_test.iloc[shap_indices]
            bad_review_idx = np.where(y_test_shap == 0)[0]
            
            if len(bad_review_idx) > 0:
                sample_idx = bad_review_idx[0]
                sample_label = "review_ruim"
            else:
                sample_idx = 0
                sample_label = "primeiro_exemplo"
            
            if hasattr(shap, 'Explanation'):
                # Para classificação binária, usar apenas a classe positiva (índice 1)
                if len(shap_to_plot.shape) == 3:  # (samples, features, classes)
                    sample_values = shap_to_plot[sample_idx, :, 1]  # Amostra específica, classe 1
                    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                else:
                    sample_values = shap_to_plot[sample_idx]  # Amostra específica
                    base_value = explainer.expected_value if not isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value[0]
                    
                shap.plots.waterfall(shap.Explanation(values=sample_values,
                                                     base_values=base_value,
                                                     data=X_test_trans_shap[sample_idx],
                                                     feature_names=fnames))
                plt.title(f'SHAP Waterfall - Explicação Local ({sample_label})')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'shap_waterfall_{sample_label}.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"[OK] SHAP waterfall plot salvo: {os.path.join(output_dir, f'shap_waterfall_{sample_label}.png')}")
            else:
                print("SHAP Explanation não disponível, pulando waterfall plot")
        except Exception as e:
            print(f"[WARN] Erro ao gerar waterfall plot: {e}")
            
    else:
        print("[WARN] SHAP analysis pulada devido a erros anteriores")

    print("Análise SHAP concluída!")

# ---------------------------
# Learning Curve (usar inference_pipeline)
# ---------------------------
print("="*60)
print("LEARNING CURVES")
print("="*60)

train_sizes, train_scores, test_scores = learning_curve(
    inference_pipeline, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='f1', n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', label='Treino')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.plot(train_sizes, test_mean, 'o-', label='Validação')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)

plt.xlabel("Tamanho do Conjunto de Treino")
plt.ylabel("F1-score")
plt.title("Learning Curves - Detecção de Overfitting")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

# Análise de overfitting - CORREÇÃO #3: Análise quantitativa detalhada
gap = train_mean[-1] - test_mean[-1]
print(f"\nANÁLISE DE OVERFITTING:")
print(f"   Gap final treino-validação: {gap:.4f}")
print(f"   Score treino final: {train_mean[-1]:.4f}")
print(f"   Score validação final: {test_mean[-1]:.4f}")

# Critérios mais detalhados para overfitting
if gap > 0.1:
    print("[ERROR] OVERFITTING SEVERO detectado! (gap > 10%)")
elif gap > 0.05:
    print("[WARN] OVERFITTING MODERADO detectado (gap > 5%)")
elif gap > 0.02:
    print("[WARN] OVERFITTING LEVE detectado (gap > 2%)")
else:
    print("[OK] Modelo bem generalizado (gap ≤ 2%)")

# Análise da convergência
train_final_diff = train_mean[-1] - train_mean[-3]  # Últimas 3 medições
test_final_diff = test_mean[-1] - test_mean[-3]

print(f"\n ANÁLISE DE CONVERGÊNCIA:")
print(f"   Melhoria treino (últimas medições): {train_final_diff:+.4f}")
print(f"   Melhoria validação (últimas medições): {test_final_diff:+.4f}")

if abs(train_final_diff) < 0.01 and abs(test_final_diff) < 0.01:
    print("[OK] Modelo convergiu adequadamente")
else:
    print("[WARN] Modelo ainda pode melhorar com mais dados")

print(f"[OK] Learning curves salvas em {output_dir}/")

# ---------------------------
# Salvar modelo treinado - MELHORIA: Salvar modelo
# ---------------------------
print("\nSALVANDO MODELO...")
model_filename = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_inference.pkl")
joblib.dump(inference_pipeline, model_filename)
print(f"[OK] Modelo salvo: {model_filename}")

# ---------------------------
# Resumo Final dos Resultados
# ---------------------------
print("="*60)
print("RESUMO FINAL DOS RESULTADOS")
print("="*60)

print(f"{model_name.upper()} - PERFORMANCE FINAL:")
print(f"F1 Macro Cross-Validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"F1 Teste (threshold=0.5): {f1_test:.4f}")
print(f"F1 Teste (threshold={best_t:.3f}): {f1_test_optimized:.4f}")
print(f"F1 Classe 0 (otimizado): {f1_class0_optimized:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

if baseline_f1 > 0:
    improvement = ((f1_test_optimized/baseline_f1 - 1) * 100)
    print(f"Melhoria sobre baseline: {improvement:.1f}%")

# Informações sobre features utilizadas
print(f"\nFEATURES UTILIZADAS ({len(features)} total):")
print(f"   Numéricas: {len(num_cols)}")
print(f"   Categóricas: {len(cat_cols)}")
if ADVANCED_FEATURES:
    advanced_count = sum([
        'customer_state' in features,
        'product_category_name' in features,
        'distancia_vendedor_cliente_km' in features
    ])
    print(f"   Features avançadas: {advanced_count}/3")

# Salvar resultados
results_dict = {
    'model_type': model_name,
    'advanced_features': ADVANCED_FEATURES,
    'threshold_optimized': best_t,
    'f1_cv_macro': cv_scores.mean(),
    'f1_cv_std': cv_scores.std(),
    'f1_test_baseline': f1_test,
    'f1_test_optimized': f1_test_optimized,
    'f1_class0_optimized': f1_class0_optimized,
    'f1_class1_optimized': f1_class1_optimized,
    'roc_auc': roc_auc,
    'pr_auc': pr_auc,
    'overfitting_gap': gap,
    'num_features': len(features),
    'num_categorical': len(cat_cols)
}

results_df = pd.DataFrame([results_dict])
results_df.to_csv(os.path.join(output_dir, 'model_results.csv'), index=False)
print(f"\nResultados salvos em: {os.path.join(output_dir, 'model_results.csv')}")

# CORREÇÃO: Lista de arquivos gerados usando output_dir
generated_files = [
    os.path.join(output_dir, 'confusion_matrix_normalized.png'),
    os.path.join(output_dir, 'roc_curve.png'), 
    os.path.join(output_dir, 'precision_recall_curve.png'),
    os.path.join(output_dir, 'permutation_importance.png'),
    os.path.join(output_dir, 'learning_curves.png'),
    os.path.join(output_dir, 'model_results.csv'),
    model_filename
]

if SHAP_AVAILABLE:
    generated_files.extend([
        os.path.join(output_dir, 'shap_summary.png')
    ])

print(f"\nArquivos gerados:")
for file in generated_files:
    print(f"- {file}")

print("\n" + "="*60)
print("🎉 ANÁLISE COMPLETA FINALIZADA! 🎉")
print("="*60)
print("[OK] Todas as melhorias implementadas:")
print("   Outputs organizados em pasta")
print("   Modelo salvo automaticamente")
print("   Features categóricas incluídas")
print("   Distância geográfica calculada")
print("   LightGBM usado (se disponível)")
print("   F1 Macro para CV balanceado")
print("   SHAP otimizado (500 amostras)")
print("   Explicação local de casos específicos")
print("="*60)