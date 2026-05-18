"""
branch_ml_module.py — Салбар таних + Train/Test ML pipeline
Аудитын ХОУ диссертацийн сайжруулалт
"""
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, classification_report, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
except Exception:
    shap = None

# ══════════════════════════════════════════════════════════
# 1. САЛБАРЫН БҮРТГЭЛ (BRANCH REGISTRY)
# ══════════════════════════════════════════════════════════

BRANCH_REGISTRY = {
    'СУИС_Бүжиг': {
        'label': '🎭 СУИС — Бүжигний урлагийн сургууль',
        'short': 'Бүжиг',
        'file_patterns': ['ej_bujig', 'бүжиг'],
        'org': 'СУИС',
    },
    'СУИС_ДУА': {
        'label': '🎨 СУИС — Дүрслэх урлагийн академи',
        'short': 'ДУА',
        'file_patterns': ['ej_dur', 'дүрслэх'],
        'org': 'СУИС',
    },
    'СУИС_ХУС': {
        'label': '🎵 СУИС — Хөгжмийн урлагийн сургууль',
        'short': 'ХУС',
        'file_patterns': ['ej_hys', 'хөгжм'],
        'org': 'СУИС',
    },
    'СУИС_КТМУС': {
        'label': '🎬 СУИС — Кино телевизийн сургууль',
        'short': 'КТМУС',
        'file_patterns': ['ej_ktmus', 'кино'],
        'org': 'СУИС',
    },
    'СУИС_ТУС': {
        'label': '🎪 СУИС — Театрын урлагийн сургууль',
        'short': 'ТУС',
        'file_patterns': ['ej_tus', 'театр'],
        'org': 'СУИС',
    },
    'СУИС_ТУД': {
        'label': '📒 СУИС — Тусгай данс',
        'short': 'ТУД',
        'file_patterns': ['ej_tud'],
        'org': 'СУИС',
    },
    'СУИС_Бусад': {
        'label': '📋 СУИС — Бусад/Нэмэлт',
        'short': 'Бусад',
        'file_patterns': ['ej_bad'],
        'org': 'СУИС',
    },
    'Дулааны_IV': {
        'label': '🏭 Дулааны IV цахилгаан станц ТӨХК',
        'short': 'Дулааны IV',
        'file_patterns': ['гүйлгээ_баланс', 'гуйлгээ баланс', 'дулааны', 'general_account_report'],
        'org': 'Дулааны IV цахилгаан станц',
    },
    'Монгол_Шуудан': {
        'label': '📮 Монгол Шуудан Компани',
        'short': 'МШК',
        'file_patterns': ['монгол шуудан', '20251015', 'нээгээд хараа'],
        'org': 'Монгол Шуудан',
    },
    'БОАЖГ': {
        'label': '🌿 БОАЖГ (Байгаль орчин)',
        'short': 'БОАЖГ',
        'file_patterns': ['боажг', 'гса_боажг', 'гса боажг'],
        'org': 'БОАЖГ',
    },
    'БЗДЕМТ': {
        'label': '🏢 БЗДЕМТ',
        'short': 'БЗДЕМТ',
        'file_patterns': ['bzdemt', 'бздемт', 'journalbzdemt'],
        'org': 'БЗДЕМТ',
    },
    '2021_Бусад': {
        'label': '📁 2021 оны бусад файлууд',
        'short': '2021',
        'file_patterns': ['2021_он', '2021 он', 'нэмэлт_санхүүжилт'],
        'org': 'Бусад (2021)',
    },
}


def detect_branch(filename, content_hint=''):
    """Файлын нэр болон агуулгаас салбарыг таних (сайжруулсан fuzzy matching)."""
    name_lower = filename.lower().replace('-', ' ').replace('_', ' ').replace('.', ' ')
    
    # Файлын нэрээр хайх
    for branch_id, info in BRANCH_REGISTRY.items():
        for pattern in info['file_patterns']:
            pattern_l = pattern.lower().replace('_', ' ')
            # Exact substring match
            if pattern_l in name_lower:
                return branch_id, info['label']
            # Pattern tokens бүгд файлын нэрэнд байвал match
            tokens = pattern_l.split()
            if len(tokens) > 1 and all(tok in name_lower for tok in tokens):
                return branch_id, info['label']
    
    # Агуулга hint-аар хайх
    if content_hint:
        hint_lower = content_hint.lower().replace('_', ' ')
        for branch_id, info in BRANCH_REGISTRY.items():
            for pattern in info['file_patterns']:
                if pattern.lower().replace('_', ' ') in hint_lower:
                    return branch_id, info['label']
    
    # Нэмэлт keyword-д суурилсан таних
    keyword_map = {
        'суис': 'СУИС_Бусад', 'бүжиг': 'СУИС_Бүжиг', 'дүрслэх': 'СУИС_ДУА',
        'хөгжим': 'СУИС_ХУС', 'кино': 'СУИС_КТМУС', 'телевиз': 'СУИС_КТМУС',
        'театр': 'СУИС_ТУС', 'дулаан': 'Дулааны_IV', 'цахилгаан станц': 'Дулааны_IV',
        'шуудан': 'Монгол_Шуудан', 'байгаль орчин': 'БОАЖГ', 'боажг': 'БОАЖГ',
        'bzdemt': 'БЗДЕМТ', 'бздемт': 'БЗДЕМТ',
        'general account': 'Дулааны_IV', 'trial balance': 'Дулааны_IV',
        'еж2025': 'Дулааны_IV', 'едт': 'Дулааны_IV',
    }
    for kw, bid in keyword_map.items():
        if kw in name_lower:
            info = BRANCH_REGISTRY.get(bid, {'label': bid})
            return bid, info.get('label', bid)
    
    return 'Тодорхойгүй', '❓ Тодорхойгүй байгууллага'


def get_branch_summary(branch_data):
    """Салбар бүрийн нэгтгэсэн статистик."""
    summary = {}
    for branch_id, df in branch_data.items():
        if df.empty:
            continue
        info = BRANCH_REGISTRY.get(branch_id, {'label': branch_id, 'short': branch_id, 'org': ''})
        n_rows = len(df)
        n_accounts = df['account_code'].nunique() if 'account_code' in df.columns else 0
        total_debit = pd.to_numeric(df.get('debit_mnt', 0), errors='coerce').fillna(0).abs().sum()
        total_credit = pd.to_numeric(df.get('credit_mnt', 0), errors='coerce').fillna(0).abs().sum()
        summary[branch_id] = {
            'Салбар': info['label'],
            'Байгууллага': info.get('org', ''),
            'Нийт мөр': n_rows,
            'Дансны тоо': n_accounts,
            'Нийт дебит': total_debit,
            'Нийт кредит': total_credit,
        }
    return pd.DataFrame(summary.values())


# ══════════════════════════════════════════════════════════
# 2. TRAIN/TEST SPLIT + SUPERVISED ML PIPELINE
# ══════════════════════════════════════════════════════════

def create_pseudo_labels(df, feat_cols, contamination=0.05):
    """Unsupervised ensemble-ийн үр дүнг pseudo-label болгон ашиглана.
    
    3 хурдан алгоритм: Isolation Forest + KMeans + Z-score
    (LOF, OneClassSVM хассан — удаан, нэмэлт чанар бага)
    Санал >= 2 бол anomaly=1.
    """
    X = df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0).astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X) if len(X) > 1 else X.values
    
    votes = np.zeros(len(df))
    
    # 1. Isolation Forest (хамгийн чухал — MCDM 21/25)
    try:
        iso = IsolationForest(contamination=min(max(contamination, 0.01), 0.40), 
                              random_state=42, n_estimators=150)
        votes += (iso.fit_predict(X) == -1).astype(int)
    except: pass
    
    # 2. KMeans distance
    try:
        k = max(2, min(8, len(df)-1 if len(df) > 2 else 2))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(Xs)
        km_dist = km.transform(Xs).min(axis=1)
        km_cut = np.percentile(km_dist, max(80, int((1-contamination)*100)))
        votes += (km_dist >= km_cut).astype(int)
    except: pass
    
    # 3. Z-score
    try:
        zmax = np.abs(Xs).max(axis=1)
        votes += (zmax > 2.5).astype(int)
    except: pass
    
    labels = (votes >= 2).astype(int)
    return labels, votes


def run_train_test_ml(df, feat_cols, test_size=0.2, contamination=0.05, random_state=42):
    """
    Бүрэн ML pipeline:
    1. Pseudo-label үүсгэх (unsupervised ensemble)
    2. Train/Test хуваах (stratified)
    3. Supervised загварууд сургах
    4. Cross-validation
    5. Гүйцэтгэлийн бүх метрик тооцох
    
    Returns: dict with all results
    """
    results = {
        'success': False,
        'error': '',
        'train_df': pd.DataFrame(),
        'test_df': pd.DataFrame(),
        'model_metrics': pd.DataFrame(),
        'cv_metrics': pd.DataFrame(),
        'roc_data': {},
        'confusion_matrices': {},
        'feature_importance': pd.DataFrame(),
        'best_model_name': '',
        'predictions': pd.DataFrame(),
    }
    
    if df is None or df.empty or len(df) < 30:
        results['error'] = f'Хангалтгүй мөр: {len(df) if df is not None else 0}. Дор хаяж 30 мөр шаардлагатай.'
        return results
    
    # Feature матриц бэлтгэх
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0
    X = df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0).astype(float)
    
    # Step 1: Pseudo-labels
    pseudo_labels, vote_counts = create_pseudo_labels(df, feat_cols, contamination)
    df = df.copy()
    df['pseudo_label'] = pseudo_labels
    df['vote_count'] = vote_counts
    
    n_pos = int(pseudo_labels.sum())
    n_neg = int(len(pseudo_labels) - n_pos)
    if n_pos < 5 or n_neg < 5:
        results['error'] = f'Pseudo-label тэнцвэргүй: anomaly={n_pos}, normal={n_neg}. Contamination-г тохируулна уу.'
        return results
    
    # Step 2: Stratified Train/Test Split
    try:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, pseudo_labels, df.index.values,
            test_size=test_size, random_state=random_state, stratify=pseudo_labels
        )
    except Exception as e:
        results['error'] = f'Train/Test хуваалт амжилтгүй: {e}'
        return results
    
    train_df = df.iloc[idx_train].copy()
    test_df = df.iloc[idx_test].copy()
    results['train_df'] = train_df
    results['test_df'] = test_df
    
    # Step 3: Supervised загварууд
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight='balanced', max_depth=10
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, random_state=42, max_depth=5, learning_rate=0.1
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced', C=1.0
        ),
    }
    
    model_metrics = []
    cv_metrics = []
    roc_data = {}
    confusion_matrices = {}
    all_importances = []
    best_f1 = -1
    best_model_name = ''
    
    for name, model in models.items():
        try:
            # Сургалт
            model.fit(X_train, y_train)
            
            # Тест дээрх таамаглал
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Test метрикүүд
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None and len(np.unique(y_test)) > 1 else 0.0
            
            model_metrics.append({
                'Загвар': name,
                'Нарийвчлал (Accuracy)': round(acc, 4),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'F1-Score': round(f1, 4),
                'AUC-ROC': round(auc, 4),
                'Сургалтын мөр': len(X_train),
                'Тестийн мөр': len(X_test),
                'Anomaly (train)': int(y_train.sum()),
                'Anomaly (test)': int(y_test.sum()),
            })
            
            # ROC curve data
            if y_proba is not None and len(np.unique(y_test)) > 1:
                fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                roc_data[name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': auc}
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            confusion_matrices[name] = cm
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
            elif hasattr(model, 'coef_'):
                imp = np.abs(model.coef_[0])
            else:
                imp = np.zeros(len(feat_cols))
            
            for i, col in enumerate(feat_cols):
                all_importances.append({
                    'Загвар': name,
                    'Шинж чанар': col,
                    'Ач холбогдол': imp[i] if i < len(imp) else 0
                })
            
            # Best model tracking
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
            
            # Step 4: Cross-validation (сургалтын өгөгдөл дээр)
            try:
                n_splits = min(5, min(int(y_train.sum()), int((1-y_train).sum())))
                if n_splits >= 2:
                    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    cv_pred = cross_val_predict(model, X_train, y_train, cv=cv)
                    cv_proba = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    cv_prec = precision_score(y_train, cv_pred, zero_division=0)
                    cv_rec = recall_score(y_train, cv_pred, zero_division=0)
                    cv_f1 = f1_score(y_train, cv_pred, zero_division=0)
                    cv_auc = roc_auc_score(y_train, cv_proba) if cv_proba is not None and len(np.unique(y_train)) > 1 else 0.0
                    
                    cv_metrics.append({
                        'Загвар': name,
                        'CV Folds': n_splits,
                        'CV Precision': round(cv_prec, 4),
                        'CV Recall': round(cv_rec, 4),
                        'CV F1-Score': round(cv_f1, 4),
                        'CV AUC-ROC': round(cv_auc, 4),
                    })
            except Exception:
                pass
            
            # Тест өгөгдөл дээрх таамаглал хадгалах
            test_df[f'{name}_pred'] = y_pred
            if y_proba is not None:
                test_df[f'{name}_proba'] = y_proba
                
        except Exception as e:
            model_metrics.append({
                'Загвар': name,
                'Нарийвчлал (Accuracy)': 0,
                'Precision': 0,
                'Recall': 0,
                'F1-Score': 0,
                'AUC-ROC': 0,
                'Алдаа': str(e),
            })
    
    # SHAP тайлбарлагч (хамгийн сайн загвар дээр)
    shap_importance = pd.DataFrame()
    if shap is not None and best_model_name == 'Random Forest' and n_pos > 10:
        try:
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)
            explainer = shap.TreeExplainer(best_model)
            shap_vals = explainer.shap_values(X_test)
            if isinstance(shap_vals, list):
                sv = shap_vals[-1]
            else:
                sv = shap_vals
            shap_abs = np.abs(sv).mean(axis=0)
            shap_importance = pd.DataFrame({
                'Шинж чанар': feat_cols,
                'SHAP ач холбогдол': shap_abs
            }).sort_values('SHAP ач холбогдол', ascending=False)
        except Exception:
            pass
    
    results['success'] = True
    results['model_metrics'] = pd.DataFrame(model_metrics)
    results['cv_metrics'] = pd.DataFrame(cv_metrics) if cv_metrics else pd.DataFrame()
    results['roc_data'] = roc_data
    results['confusion_matrices'] = confusion_matrices
    results['feature_importance'] = pd.DataFrame(all_importances) if all_importances else pd.DataFrame()
    results['shap_importance'] = shap_importance
    results['best_model_name'] = best_model_name
    results['test_df'] = test_df
    results['pseudo_label_stats'] = {
        'total': len(pseudo_labels),
        'anomaly': int(pseudo_labels.sum()),
        'normal': int(len(pseudo_labels) - pseudo_labels.sum()),
        'anomaly_pct': round(pseudo_labels.mean() * 100, 2),
    }
    
    return results


def run_branch_comparison(branch_data, feat_cols, contamination=0.05):
    """Салбар бүрийн ML үр дүнг нэгтгэж харьцуулна."""
    comparison = []
    branch_anomalies = {}
    
    for branch_id, df in branch_data.items():
        if df.empty or len(df) < 20:
            continue
        
        info = BRANCH_REGISTRY.get(branch_id, {'label': branch_id, 'short': branch_id})
        
        # Feature бэлтгэх
        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0
        X = df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0).astype(float)
        
        # Pseudo-label
        labels, votes = create_pseudo_labels(df, feat_cols, contamination)
        
        n_anomaly = int(labels.sum())
        pct_anomaly = round(labels.mean() * 100, 2)
        
        # Дундаж vote
        avg_vote = round(votes.mean(), 2)
        max_vote = int(votes.max())
        
        # Гүйлгээний статистик
        total_amount = 0
        if 'amount' in df.columns:
            total_amount = pd.to_numeric(df['amount'], errors='coerce').fillna(0).abs().sum()
        elif 'debit_mnt' in df.columns:
            total_amount = pd.to_numeric(df['debit_mnt'], errors='coerce').fillna(0).abs().sum() + \
                          pd.to_numeric(df.get('credit_mnt', 0), errors='coerce').fillna(0).abs().sum()
        
        comparison.append({
            'Салбар': info['short'],
            'Нийт мөр': len(df),
            'Аномали тоо': n_anomaly,
            'Аномали %': pct_anomaly,
            'Дундаж vote': avg_vote,
            'Max vote': max_vote,
            'Нийт дүн (сая₮)': round(total_amount / 1e6, 1),
        })
        
        # Аномали гүйлгээнүүдийг хадгалах
        df_copy = df.copy()
        df_copy['pseudo_label'] = labels
        df_copy['vote_count'] = votes
        branch_anomalies[branch_id] = df_copy[df_copy['pseudo_label'] == 1]
    
    return pd.DataFrame(comparison), branch_anomalies


# ══════════════════════════════════════════════════════════
# 3. UNSUPERVISED VS SUPERVISED ХАРЬЦУУЛАЛТ
# ══════════════════════════════════════════════════════════

def compare_unsupervised_supervised(df, feat_cols, contamination=0.05):
    """Unsupervised ensemble vs Supervised загварын харьцуулалт.
    
    Диссертацид хэрэглэхэд тохиромжтой хүснэгт гаргана.
    """
    results = {}
    
    X = df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0).astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    
    # Unsupervised загварууд тус бүрийн илрүүлэлт
    unsup_models = {}
    
    try:
        iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=250)
        unsup_models['Isolation Forest'] = (iso.fit_predict(X) == -1).astype(int)
    except: pass
    
    try:
        lof = LocalOutlierFactor(n_neighbors=max(5, min(20, len(df)-1)),
                                  contamination=min(max(contamination, 0.01), 0.40))
        unsup_models['LOF'] = (lof.fit_predict(Xs) == -1).astype(int)
    except: pass
    
    try:
        svm = OneClassSVM(nu=min(max(contamination, 0.01), 0.40), kernel='rbf', gamma='scale')
        svm.fit(Xs)
        unsup_models['One-Class SVM'] = (svm.predict(Xs) == -1).astype(int)
    except: pass
    
    try:
        km = KMeans(n_clusters=max(2, min(8, len(df)-1)), random_state=42, n_init=10)
        km.fit(Xs)
        km_dist = km.transform(Xs).min(axis=1)
        km_cut = np.percentile(km_dist, max(80, int((1-contamination)*100)))
        unsup_models['KMeans'] = (km_dist >= km_cut).astype(int)
    except: pass
    
    try:
        zmax = np.abs(Xs).max(axis=1)
        unsup_models['Z-score'] = (zmax > 2.8).astype(int)
    except: pass
    
    # Ensemble label
    if unsup_models:
        vote_matrix = np.column_stack(list(unsup_models.values()))
        ensemble_label = (vote_matrix.sum(axis=1) >= 2).astype(int)
        unsup_models['Ensemble (≥2)'] = ensemble_label
    
    # Unsupervised тойм
    unsup_summary = []
    for name, preds in unsup_models.items():
        unsup_summary.append({
            'Загвар': name,
            'Төрөл': 'Unsupervised',
            'Илрүүлсэн аномали': int(preds.sum()),
            'Аномали %': round(preds.mean() * 100, 2),
        })
    
    results['unsupervised_summary'] = pd.DataFrame(unsup_summary)
    results['unsupervised_models'] = unsup_models
    
    return results


# ══════════════════════════════════════════════════════════
# 4. ҮЕ ШАТ 3: САЛБАР ХООРОНДЫН PATTERN DETECTION
# ══════════════════════════════════════════════════════════

def run_cross_branch_patterns(branch_engineered, feat_cols, contamination=0.05):
    """Салбар дундын хэв маяг илрүүлэх.
    - Бүх салбарт давтагддаг аномали төрөл
    - Салбарт өвөрмөц аномали
    - Feature бүрийн салбар хоорондын ялгаа
    """
    results = {
        'common_patterns': [],
        'unique_patterns': [],
        'feature_by_branch': pd.DataFrame(),
        'anomaly_type_matrix': pd.DataFrame(),
        'branch_correlation': pd.DataFrame(),
    }
    
    if not branch_engineered or len(branch_engineered) < 2:
        return results
    
    # Feature дундаж утга салбар бүрээр
    feature_means = {}
    branch_anomaly_profiles = {}
    
    for bid, df in branch_engineered.items():
        if df.empty or len(df) < 10:
            continue
        info = BRANCH_REGISTRY.get(bid, {'short': bid})
        
        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0
        
        labels, votes = create_pseudo_labels(df, feat_cols, contamination)
        df_copy = df.copy()
        df_copy['anomaly'] = labels
        
        # Feature дундаж (аномали vs хэвийн)
        anom_means = df_copy[df_copy['anomaly']==1][feat_cols].mean()
        normal_means = df_copy[df_copy['anomaly']==0][feat_cols].mean()
        
        row = {'Салбар': info['short']}
        for f in feat_cols:
            row[f'{f}_anom'] = round(anom_means.get(f, 0), 4)
            row[f'{f}_norm'] = round(normal_means.get(f, 0), 4)
            row[f'{f}_diff'] = round(anom_means.get(f, 0) - normal_means.get(f, 0), 4)
        feature_means[bid] = row
        
        # Аномали профайл: аль feature хамгийн их нөлөөтэй
        diffs = {f: abs(anom_means.get(f, 0) - normal_means.get(f, 0)) for f in feat_cols}
        top3 = sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:3]
        branch_anomaly_profiles[bid] = {
            'branch': info['short'],
            'total': len(df_copy),
            'anomaly_count': int(labels.sum()),
            'anomaly_pct': round(labels.mean() * 100, 2),
            'top_feature_1': top3[0][0] if len(top3) > 0 else '',
            'top_feature_2': top3[1][0] if len(top3) > 1 else '',
            'top_feature_3': top3[2][0] if len(top3) > 2 else '',
        }
    
    results['feature_by_branch'] = pd.DataFrame(feature_means.values())
    
    # Аномали төрлийн матриц (салбар × feature flag)
    type_matrix = []
    for bid, profile in branch_anomaly_profiles.items():
        type_matrix.append(profile)
    results['anomaly_type_matrix'] = pd.DataFrame(type_matrix)
    
    # Common patterns: бүх салбарт top feature ижил
    if branch_anomaly_profiles:
        all_top1 = [v['top_feature_1'] for v in branch_anomaly_profiles.values() if v['top_feature_1']]
        from collections import Counter
        top1_counts = Counter(all_top1)
        n_branches = len(branch_anomaly_profiles)
        for feat, count in top1_counts.items():
            if count >= max(2, n_branches * 0.5):
                results['common_patterns'].append({
                    'Хэв маяг': f'{feat} — бүх салбарын аномалид давамгайлж байна',
                    'Давтамж': f'{count}/{n_branches} салбар',
                    'Төрөл': 'Нийтлэг'
                })
        
        # Unique patterns: зөвхөн нэг салбарт
        all_top_feats = []
        for v in branch_anomaly_profiles.values():
            all_top_feats.extend([v['top_feature_1'], v['top_feature_2'], v['top_feature_3']])
        feat_branch_map = {}
        for bid, v in branch_anomaly_profiles.items():
            for feat_key in ['top_feature_1', 'top_feature_2', 'top_feature_3']:
                feat = v[feat_key]
                if feat:
                    feat_branch_map.setdefault(feat, []).append(v['branch'])
        for feat, branches in feat_branch_map.items():
            if len(branches) == 1:
                results['unique_patterns'].append({
                    'Хэв маяг': f'{feat} — зөвхөн {branches[0]}-д илэрсэн',
                    'Салбар': branches[0],
                    'Төрөл': 'Өвөрмөц'
                })
    
    # Салбар хоорондын корреляци (аномали хувиар)
    if len(branch_anomaly_profiles) >= 2:
        corr_data = []
        for bid, df in branch_engineered.items():
            if bid not in branch_anomaly_profiles:
                continue
            for c in feat_cols:
                if c not in df.columns:
                    df[c] = 0
            row = {f: df[f].mean() for f in feat_cols if f in df.columns}
            row['branch'] = BRANCH_REGISTRY.get(bid, {'short': bid})['short']
            corr_data.append(row)
        if corr_data:
            corr_df = pd.DataFrame(corr_data).set_index('branch')
            try:
                results['branch_correlation'] = corr_df.T.corr()
            except:
                pass
    
    return results


# ══════════════════════════════════════════════════════════
# 5. ҮЕ ШАТ 4: LEARNING CURVE + HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════

def run_learning_curve(X, y, model_class, model_params, n_points=8, cv_folds=5):
    """Сургалтын процессийн муруй (Learning curve).
    Сургалтын өгөгдлийн хэмжээг өөрчлөхөд гүйцэтгэл хэрхэн өөрчлөгдөхийг харуулна.
    """
    from sklearn.model_selection import learning_curve as sk_learning_curve
    
    try:
        model = model_class(**model_params)
        n_samples = len(X)
        min_samples = max(30, int(n_samples * 0.1))
        train_sizes = np.linspace(min_samples / n_samples, 0.95, n_points)
        
        n_splits = min(cv_folds, min(int(y.sum()), int((1-y).sum())))
        if n_splits < 2:
            return None
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        train_sizes_abs, train_scores, test_scores = sk_learning_curve(
            model, X, y, train_sizes=train_sizes, cv=cv,
            scoring='f1', n_jobs=1, random_state=42
        )
        
        result = pd.DataFrame({
            'train_size': train_sizes_abs,
            'train_f1_mean': train_scores.mean(axis=1),
            'train_f1_std': train_scores.std(axis=1),
            'test_f1_mean': test_scores.mean(axis=1),
            'test_f1_std': test_scores.std(axis=1),
        })
        return result
    except Exception as e:
        return None


def run_hyperparameter_search(X_train, y_train, X_test, y_test, model_name='Random Forest'):
    """Hyperparameter тохиргооны хайлт.
    Загвар бүрийн гол параметрүүдийг grid search-ээр тохируулна.
    """
    results = []
    
    if model_name == 'Random Forest':
        param_grid = [
            {'n_estimators': n, 'max_depth': d}
            for n in [50, 100, 200, 300]
            for d in [5, 8, 10, 15, None]
        ]
        for params in param_grid:
            try:
                model = RandomForestClassifier(
                    random_state=42, class_weight='balanced', **params
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                results.append({
                    'n_estimators': params['n_estimators'],
                    'max_depth': str(params['max_depth']),
                    'F1': round(f1, 4),
                    'AUC': round(auc, 4),
                    'Precision': round(prec, 4),
                    'Recall': round(rec, 4),
                })
            except:
                pass
    
    elif model_name == 'Gradient Boosting':
        param_grid = [
            {'n_estimators': n, 'max_depth': d, 'learning_rate': lr}
            for n in [50, 100, 200]
            for d in [3, 5, 7]
            for lr in [0.01, 0.05, 0.1, 0.2]
        ]
        for params in param_grid:
            try:
                model = GradientBoostingClassifier(random_state=42, **params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0
                results.append({
                    'n_estimators': params['n_estimators'],
                    'max_depth': params['max_depth'],
                    'learning_rate': params['learning_rate'],
                    'F1': round(f1, 4),
                    'AUC': round(auc, 4),
                })
            except:
                pass
    
    elif model_name == 'Logistic Regression':
        for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
            try:
                model = LogisticRegression(
                    C=C, max_iter=2000, random_state=42, class_weight='balanced'
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0
                results.append({
                    'C': C,
                    'F1': round(f1, 4),
                    'AUC': round(auc, 4),
                })
            except:
                pass
    
    return pd.DataFrame(results)


def run_stability_analysis(X, y, feat_cols, n_runs=10, test_size=0.2):
    """Загварын тогтвортой байдлын шинжилгээ.
    Олон удаа train/test хуваалт хийж, метрикүүдийн тархалтыг шалгана.
    """
    from sklearn.model_selection import train_test_split
    
    models_config = {
        'Random Forest': lambda: RandomForestClassifier(n_estimators=200, random_state=None, class_weight='balanced', max_depth=10),
        'Gradient Boosting': lambda: GradientBoostingClassifier(n_estimators=150, random_state=None, max_depth=5),
        'Logistic Regression': lambda: LogisticRegression(max_iter=1000, random_state=None, class_weight='balanced'),
    }
    
    all_results = []
    
    for run_i in range(n_runs):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=run_i * 7 + 13, stratify=y
        )
        for name, model_fn in models_config.items():
            try:
                model = model_fn()
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None
                f1 = f1_score(y_te, y_pred, zero_division=0)
                auc = roc_auc_score(y_te, y_proba) if y_proba is not None and len(np.unique(y_te)) > 1 else 0
                prec = precision_score(y_te, y_pred, zero_division=0)
                rec = recall_score(y_te, y_pred, zero_division=0)
                all_results.append({
                    'Run': run_i + 1,
                    'Загвар': name,
                    'Precision': prec,
                    'Recall': rec,
                    'F1': f1,
                    'AUC': auc,
                })
            except:
                pass
    
    df = pd.DataFrame(all_results)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Тогтвортой байдлын нэгтгэл
    summary = df.groupby('Загвар').agg(
        F1_дундаж=('F1', 'mean'),
        F1_std=('F1', 'std'),
        AUC_дундаж=('AUC', 'mean'),
        AUC_std=('AUC', 'std'),
        Prec_дундаж=('Precision', 'mean'),
        Prec_std=('Precision', 'std'),
        Recall_дундаж=('Recall', 'mean'),
        Recall_std=('Recall', 'std'),
    ).round(4).reset_index()
    
    # CV (coefficient of variation)
    summary['F1_CV%'] = (summary['F1_std'] / summary['F1_дундаж'].replace(0, np.nan) * 100).round(2).fillna(0)
    summary['AUC_CV%'] = (summary['AUC_std'] / summary['AUC_дундаж'].replace(0, np.nan) * 100).round(2).fillna(0)
    
    return df, summary


def run_mcnemar_test(y_true, pred_a, pred_b, model_a_name='Model A', model_b_name='Model B'):
    """McNemar тест: хоёр загварын таамаглалын ялгааны статистик ач холбогдол.
    
    H0: Хоёр загварын алдааны хэв маяг ижил
    H1: Хоёр загварын алдааны хэв маяг ялгаатай
    """
    # Contingency table
    # a: both correct, b: A correct B wrong, c: A wrong B correct, d: both wrong
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    
    b = int((correct_a & ~correct_b).sum())  # A зөв, B буруу
    c = int((~correct_a & correct_b).sum())  # A буруу, B зөв
    
    # McNemar statistic (continuity correction)
    if b + c == 0:
        return {
            'test': 'McNemar',
            'model_a': model_a_name,
            'model_b': model_b_name,
            'b (A✓ B✗)': b,
            'c (A✗ B✓)': c,
            'statistic': 0,
            'p_value': 1.0,
            'тайлбар': 'Хоёр загвар бүрэн ижил таамагласан'
        }
    
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(stat, df=1)
    
    if p_value < 0.01:
        interp = 'Маш өндөр ач холбогдолтой ялгаа (p<0.01)'
    elif p_value < 0.05:
        interp = 'Ач холбогдолтой ялгаа (p<0.05)'
    elif p_value < 0.10:
        interp = 'Сул ялгаа (p<0.10)'
    else:
        interp = 'Статистик ач холбогдолтой ялгаа алга (p≥0.10)'
    
    return {
        'Model A': model_a_name,
        'Model B': model_b_name,
        'b (A✓ B✗)': b,
        'c (A✗ B✓)': c,
        'McNemar χ²': round(stat, 4),
        'p-value': round(p_value, 6),
        'Дүгнэлт': interp
    }


def run_contamination_sensitivity(X, y_pseudo_fn, feat_cols, test_size=0.2):
    """Contamination параметрийн мэдрэмжийн шинжилгээ.
    Өөр өөр contamination утга дээр загварын гүйцэтгэлийг харьцуулна.
    """
    from sklearn.model_selection import train_test_split
    
    results = []
    for cont in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
        try:
            labels, _ = y_pseudo_fn(cont)
            n_pos = int(labels.sum())
            n_neg = int(len(labels) - n_pos)
            if n_pos < 5 or n_neg < 5:
                continue
            
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, labels, test_size=test_size, random_state=42, stratify=labels
            )
            
            rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', max_depth=10)
            rf.fit(X_tr, y_tr)
            y_pred = rf.predict(X_te)
            y_proba = rf.predict_proba(X_te)[:, 1]
            
            results.append({
                'Contamination': cont,
                'Аномали тоо': n_pos,
                'Аномали %': round(labels.mean()*100, 2),
                'F1': round(f1_score(y_te, y_pred, zero_division=0), 4),
                'AUC': round(roc_auc_score(y_te, y_proba), 4) if len(np.unique(y_te)) > 1 else 0,
                'Precision': round(precision_score(y_te, y_pred, zero_division=0), 4),
                'Recall': round(recall_score(y_te, y_pred, zero_division=0), 4),
            })
        except:
            pass
    
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════
# 6. ҮЕ ШАТ 5: ДИССЕРТАЦИЙН ХҮСНЭГТҮҮД
# ══════════════════════════════════════════════════════════

def generate_dissertation_tables(ml_results, branch_comparison=None, stability=None,
                                  mcnemar_results=None, hp_results=None):
    """Диссертацид шаардлагатай бүх хүснэгтийг нэгтгэн үүсгэнэ.
    Returns: dict of (table_name -> DataFrame)
    """
    tables = {}
    
    # Хүснэгт 1: Загварын гүйцэтгэлийн нэгтгэл
    if ml_results and ml_results.get('success'):
        mm = ml_results.get('model_metrics', pd.DataFrame())
        if not mm.empty:
            t1 = mm[['Загвар','Нарийвчлал (Accuracy)','Precision','Recall','F1-Score','AUC-ROC']].copy()
            t1.columns = ['Загвар','Accuracy','Precision','Recall','F1-Score','AUC-ROC']
            tables['Хүснэгт_1_Загварын_гүйцэтгэл'] = t1
    
    # Хүснэгт 2: Cross-validation
    if ml_results and ml_results.get('success'):
        cv = ml_results.get('cv_metrics', pd.DataFrame())
        if cv is not None and not cv.empty:
            tables['Хүснэгт_2_Cross_validation'] = cv
    
    # Хүснэгт 3: Pseudo-label статистик
    if ml_results and ml_results.get('success'):
        ps = ml_results.get('pseudo_label_stats', {})
        if ps:
            tables['Хүснэгт_3_Pseudo_label'] = pd.DataFrame([{
                'Нийт гүйлгээ': ps.get('total', 0),
                'Аномали': ps.get('anomaly', 0),
                'Хэвийн': ps.get('normal', 0),
                'Аномали %': ps.get('anomaly_pct', 0),
                'Сургалт': len(ml_results.get('train_df', [])),
                'Тест': len(ml_results.get('test_df', [])),
            }])
    
    # Хүснэгт 4: Feature importance
    if ml_results and ml_results.get('success'):
        fi = ml_results.get('feature_importance', pd.DataFrame())
        if fi is not None and not fi.empty:
            # Загвар бүрийн top features
            pivot = fi.pivot_table(index='Шинж чанар', columns='Загвар', values='Ач холбогдол').round(4)
            pivot['Дундаж'] = pivot.mean(axis=1).round(4)
            pivot = pivot.sort_values('Дундаж', ascending=False)
            tables['Хүснэгт_4_Feature_importance'] = pivot.reset_index()
    
    # Хүснэгт 5: Салбарын харьцуулалт
    if branch_comparison is not None and not branch_comparison.empty:
        tables['Хүснэгт_5_Салбарын_харьцуулалт'] = branch_comparison
    
    # Хүснэгт 6: Тогтвортой байдал
    if stability is not None and not stability.empty:
        tables['Хүснэгт_6_Тогтвортой_байдал'] = stability
    
    # Хүснэгт 7: McNemar тестүүд
    if mcnemar_results:
        tables['Хүснэгт_7_McNemar_тест'] = pd.DataFrame(mcnemar_results)
    
    # Хүснэгт 8: Hyperparameter tuning
    if hp_results is not None and not hp_results.empty:
        tables['Хүснэгт_8_Hyperparameter'] = hp_results
    
    return tables


# ══════════════════════════════════════════════════════════
# 7. ИЛРҮҮЛЭЛТИЙН ЭРСДЭЛИЙН ТООЦОО (DR = 1 - Recall)
# ══════════════════════════════════════════════════════════

def compute_detection_risk(ml_results, mus_coverage=0.20):
    """DR = 1 - Recall тооцоо. AI vs MUS харьцуулалт.
    
    Returns: DataFrame with DR comparison per model + MUS baseline
    """
    if not ml_results or not ml_results.get('success'):
        return pd.DataFrame()
    
    mm = ml_results.get('model_metrics', pd.DataFrame())
    if mm.empty:
        return pd.DataFrame()
    
    rows = []
    for _, row in mm.iterrows():
        recall = row.get('Recall', 0)
        dr_ai = round((1 - recall) * 100, 2)
        dr_mus = round((1 - mus_coverage) * 100, 2)  # MUS 20% → DR 80% (conservative)
        improvement = round(dr_mus / dr_ai, 1) if dr_ai > 0 else float('inf')
        
        rows.append({
            'Загвар': row['Загвар'],
            'Recall': round(recall, 4),
            'AI DR (%)': dr_ai,
            'MUS DR (%)': dr_mus,
            'Сайжруулалт (дахин)': improvement,
            'ISA 200 шаардлага (<5%)': '✓' if dr_ai < 5 else '✗',
        })
    
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
# 8. ISA СТАНДАРТ ↔ ШИНЖ ЧАНАРЫН MAPPING
# ══════════════════════════════════════════════════════════

ISA_FEATURE_MAP = {
    'ISA 200 (Аудитын эрсдэл)': {
        'features': ['log_amount', 'amt_zscore'],
        'description': 'AR = IR × CR × DR. DR = 1 − Recall',
    },
    'ISA 240 (Залилан)': {
        'features': ['is_round', 'is_dup', 'pair_rare', 'cp_rare'],
        'description': 'Дугуй тоо, давхардсан гүйлгээ, ховор харилцагч',
    },
    'ISA 315 (Эрсдэл тодорхойлох)': {
        'features': ['acct_cat_num', 'is_debit'],
        'description': 'Дансны ангилал, дебит/кредит чиглэл',
    },
    'ISA 500 (Нотолгоо)': {
        'features': ['desc_mismatch', 'name_no_overlap', 'dir_mismatch', 'desc_empty'],
        'description': 'Тайлбар зөрүү, нэр таарахгүй, чиглэл зөрүү',
    },
    'ISA 520 (Аналитик горим)': {
        'features': ['log_amount', 'benford_dev', 'amt_zscore'],
        'description': 'Он дамнасан өөрчлөлт, Бенфордын хазайлт',
    },
    'ISA 530 (Түүвэрлэлт)': {
        'features': [],
        'description': '100% хамрах хүрээ (MUS 20%-ийг орлосон)',
    },
}

def get_isa_feature_report(feature_importance_df):
    """ISA стандарт бүрийн шинж чанарын ач холбогдлыг нэгтгэнэ."""
    if feature_importance_df is None or feature_importance_df.empty:
        return pd.DataFrame()
    
    rows = []
    for isa, info in ISA_FEATURE_MAP.items():
        if not info['features']:
            rows.append({
                'ISA Стандарт': isa, 'Тайлбар': info['description'],
                'Дундаж ач холбогдол': '-', 'Шинж чанарууд': '-',
            })
            continue
        
        mask = feature_importance_df['Шинж чанар'].isin(info['features'])
        matched = feature_importance_df[mask]
        if not matched.empty:
            avg_imp = matched['Ач холбогдол'].mean()
            feat_list = ', '.join(matched['Шинж чанар'].unique())
        else:
            avg_imp = 0
            feat_list = ', '.join(info['features'])
        
        rows.append({
            'ISA Стандарт': isa, 'Тайлбар': info['description'],
            'Дундаж ач холбогдол': round(avg_imp, 4),
            'Шинж чанарууд': feat_list,
        })
    
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
# 9. БЕНФОРДЫН ШИНЖИЛГЭЭ
# ══════════════════════════════════════════════════════════

BENFORD_EXPECTED = {1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097, 5: 0.079,
                    6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046}

def run_benford_analysis(amounts):
    """Бенфордын хуулийн шинжилгээ.
    
    Returns: dict with observed, expected, chi2, p_value, deviations
    """
    if amounts is None or len(amounts) == 0:
        return None
    
    # Эхний цифрийг олох
    abs_amounts = pd.Series(amounts).abs().dropna()
    abs_amounts = abs_amounts[abs_amounts > 0]
    
    if len(abs_amounts) < 50:
        return None
    
    first_digits = abs_amounts.apply(lambda x: int(str(abs(x)).lstrip('0').lstrip('.')[0]) if x != 0 else 0)
    first_digits = first_digits[first_digits.between(1, 9)]
    
    observed_counts = first_digits.value_counts().sort_index()
    n = len(first_digits)
    
    observed = {}
    expected = {}
    deviations = {}
    
    for d in range(1, 10):
        obs_count = observed_counts.get(d, 0)
        obs_pct = obs_count / n if n > 0 else 0
        exp_pct = BENFORD_EXPECTED[d]
        observed[d] = round(obs_pct, 4)
        expected[d] = round(exp_pct, 4)
        deviations[d] = round(abs(obs_pct - exp_pct), 4)
    
    # Chi-square test
    from scipy.stats import chisquare
    obs_arr = np.array([observed_counts.get(d, 0) for d in range(1, 10)])
    exp_arr = np.array([BENFORD_EXPECTED[d] * n for d in range(1, 10)])
    
    try:
        chi2, p_value = chisquare(obs_arr, f_exp=exp_arr)
    except:
        chi2, p_value = 0, 1
    
    return {
        'observed': observed,
        'expected': expected,
        'deviations': deviations,
        'chi2': round(chi2, 4),
        'p_value': round(p_value, 6),
        'n': n,
        'max_deviation': max(deviations.values()) if deviations else 0,
        'conform': p_value > 0.05,  # True = Бенфордын хуульд нийцэж байна
    }


# ══════════════════════════════════════════════════════════
# 10. ЭРСДЭЛИЙН НЭГДСЭН ОНОО
# ══════════════════════════════════════════════════════════

def compute_risk_score(df, feat_cols):
    """Гүйлгээ бүрт 0-100 эрсдэлийн оноо тооцоолно.
    
    Олон шинж чанарын weighted sum дээр суурилсан.
    """
    score = np.zeros(len(df))
    
    weights = {
        'amt_zscore': 20, 'benford_dev': 15, 'is_round': 10,
        'is_dup': 15, 'desc_mismatch': 10, 'name_no_overlap': 10,
        'dir_mismatch': 10, 'pair_rare': 5, 'cp_rare': 5,
        'desc_empty': 5, 'log_amount': 5,
    }
    
    for col, w in weights.items():
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').fillna(0)
            # Normalize to 0-1
            vmin, vmax = vals.min(), vals.max()
            if vmax > vmin:
                normalized = (vals - vmin) / (vmax - vmin)
            else:
                normalized = vals * 0
            score += normalized.values * w
    
    # Scale to 0-100
    smin, smax = score.min(), score.max()
    if smax > smin:
        score = (score - smin) / (smax - smin) * 100
    
    return np.round(score, 1)


