import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

# Temizlenmiş veri setini yükleme
cleaned_file_path = r'C:\Users\Pc\Downloads\cleaned_diabetes_drug_interactions.csv'
data = pd.read_csv(cleaned_file_path)

# Özellikler ve Etiketler
feature_columns = data.columns[2:-1] if 'label' in data.columns else data.columns[2:]
X = pd.get_dummies(data[feature_columns])  # One-hot encoding for categorical features
y = data['Risk Seviyesi'].values if 'Risk Seviyesi' in data.columns else data.iloc[:, -1].values

# K-Fold ayarı
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Performans metriklerini saklamak için listeler
all_y_true = []
all_y_pred = []
fold_accuracies = []  # Her fold için doğruluk oranlarını saklamak için liste

# Anti-Overfitting Tekniklerini İçeren Model Pipeline
rf_model = Pipeline([
    ('scaler', StandardScaler()),  # Özellik ölçeklendirme
    ('classifier', RandomForestClassifier(random_state=42))
])

dt_model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Ensemble model
ensemble_model = VotingClassifier(
    estimators=[('random_forest', rf_model), ('decision_tree', dt_model)],
    voting='soft'
)

# Hiperparametreler için GridSearchCV'yi tanımla
param_grid_rf = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, 20],
    'classifier__min_samples_split': [2, 10, 20],
    'classifier__min_samples_leaf': [2, 5, 10]
}

param_grid_dt = {
    'classifier__max_depth': [5, 10, 20],
    'classifier__min_samples_split': [2, 10, 20],
    'classifier__min_samples_leaf': [2, 5, 10]
}

# K-Fold ile model performansını değerlendir
for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Modeli eğit (GridSearchCV ile)
    grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=3, n_jobs=-1, verbose=1)
    grid_search_dt = GridSearchCV(dt_model, param_grid_dt, cv=3, n_jobs=-1, verbose=1)

    grid_search_rf.fit(X_train, y_train)
    grid_search_dt.fit(X_train, y_train)

    # En iyi parametreleri yazdır
    print(f"Fold {fold + 1} için En İyi RandomForest Parametreleri: {grid_search_rf.best_params_}")
    print(f"Fold {fold + 1} için En İyi DecisionTree Parametreleri: {grid_search_dt.best_params_}")

    # En iyi modellerle tahmin yap
    best_rf_model = grid_search_rf.best_estimator_
    best_dt_model = grid_search_dt.best_estimator_

    # Ensemble modelini güncelle
    ensemble_model = VotingClassifier(
        estimators=[('random_forest', best_rf_model), ('decision_tree', best_dt_model)],
        voting='soft'
    )

    # Modeli eğit
    ensemble_model.fit(X_train, y_train)

    # Test seti üzerinde tahmin
    y_pred = ensemble_model.predict(X_test)

    # Tahmin sonuçlarını sakla
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    # Her fold için doğruluk oranını hesapla ve sakla
    fold_accuracy = accuracy_score(y_test, y_pred)
    fold_accuracies.append(fold_accuracy)
    print(f"Fold {fold + 1} Doğruluk Oranı: {fold_accuracy:.4f}")

# Genel performans metrikleri
print("\nK-Fold Sonuçları:")
print(f"Ortalama Doğruluk: {accuracy_score(all_y_true, all_y_pred):.4f}")
print(f"Precision: {precision_score(all_y_true, all_y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall: {recall_score(all_y_true, all_y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1-Score: {f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0):.4f}")

# Karmaşıklık Matrisi
cm = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Karmaşıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değerler')
plt.show()

# Hasta senaryosu oluşturma
class Hasta:
    def __init__(self, yas, komplikasyon, hipoglisemi_risk, yasam_beklentisi, kan_sekeri, hba1c, kardiyovaskuler_hastalik, gfr, bmi, maliyet_onemli, karaciger_yetmezligi, yeni_tani, kbh):
        self.yas = yas
        self.komplikasyon = komplikasyon
        self.hipoglisemi_risk = hipoglisemi_risk
        self.yasam_beklentisi = yasam_beklentisi
        self.kan_sekeri = kan_sekeri
        self.hba1c = hba1c
        self.kardiyovaskuler_hastalik = kardiyovaskuler_hastalik
        self.gfr = gfr
        self.bmi = bmi
        self.maliyet_onemli = maliyet_onemli
        self.karaciger_yetmezligi = karaciger_yetmezligi
        self.yeni_tani = yeni_tani
        self.kbh = kbh

# Örnek hasta senaryosu
hasta = Hasta(
    yas=70,
    komplikasyon=False,
    hipoglisemi_risk=True,
    yasam_beklentisi=10,
    kan_sekeri=320,
    hba1c=11,
    kardiyovaskuler_hastalik=True,
    gfr=40,
    bmi=32,
    maliyet_onemli=False,
    karaciger_yetmezligi=False,
    yeni_tani=False,
    kbh=True
)

# İlaç önerisi yapma fonksiyonu
def hasta_hedef_a1c(hasta):
    if (hasta.yas < 45 and not hasta.komplikasyon and
        not hasta.hipoglisemi_risk and hasta.yasam_beklentisi > 15):
        return 6.5
    elif hasta.yas > 65 or hasta.komplikasyon or hasta.hipoglisemi_risk:
        return 8.0
    else:
        return 7.0

def ilac_oncelik_belirle(hasta):
    hedef_a1c = hasta_hedef_a1c(hasta)

    # Semptomatik hiperglisemi
    if hasta.kan_sekeri > 300:
        return ["insulin"]

    # Başlangıç tedavisi
    if hasta.yeni_tani and not hasta.komplikasyon:
        return ["metformin"]

    # Ciddi hiperglisemi
    if hasta.hba1c > 10 or hasta.hba1c > (hedef_a1c + 3):
        return ["insulin", "metformin"]

    # Kardiyovasküler hastalık
    if hasta.kardiyovaskuler_hastalik:
        if hasta.gfr >= 30:
            return ["metformin", "sglt2_inhibitoru", "glp1_agonist"]
        else:
            return ["metformin", "glp1_agonist"]

    # Kronik böbrek hastalığı
    if hasta.kbh:
        if hasta.gfr >= 30:
            return ["metformin", "sglt2_inhibitoru", "glp1_agonist"]
        elif 15 <= hasta.gfr < 30:
            return ["metformin_doz_azalt", "glp1_agonist", "dpp4_inhibitoru"]
        else:
            return ["insulin"]

    # Obezite
    if hasta.bmi >= 30:
        return ["metformin", "glp1_agonist", "sglt2_inhibitoru"]

    # Hipoglisemi riski
    if hasta.hipoglisemi_risk:
        return ["metformin", "dpp4_inhibitoru", "sglt2_inhibitoru", "glp1_agonist"]

    # Yaşlı hasta
    if hasta.yas >= 65:
        return ["metformin", "dpp4_inhibitoru"]

    # Maliyet önemli
    if hasta.maliyet_onemli:
        return ["metformin", "sulfonylure"]

    # Karaciğer yetmezliği
    if hasta.karaciger_yetmezligi:
        return ["insulin", "metformin_doz_azalt"]

    # Standart tedavi
    return ["metformin", "dpp4_inhibitoru", "sglt2_inhibitoru"]

# İlaç önerisi yap
ilac_onerisi = ilac_oncelik_belirle(hasta)
print("Önerilen İlaçlar:", ilac_onerisi)
