import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from modelo_xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score

# Load the competition data
comp_data = pd.read_csv("competition_data.csv")
comp_data["date"] = pd.to_datetime(comp_data["date"])

# Divide la columna de fecha en día, mes y año
comp_data['day'] = comp_data['date'].dt.day
comp_data['month'] = comp_data['date'].dt.month
comp_data['year'] = comp_data['date'].dt.year

# Split into training and evaluation samples
train_data = comp_data[comp_data["ROW_ID"].isna()]
eval_data = comp_data[comp_data["ROW_ID"].notna()]
del comp_data
gc.collect()

# Dividir los datos en conjunto de entrenamiento y prueba (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    train_data.drop(columns=["conversion", "ROW_ID"]).select_dtypes(include='number'),
    train_data["conversion"],
    test_size=0.3,  # Proporción para el conjunto de prueba
    random_state=42  # Semilla aleatoria para reproducibilidad
)
del train_data
gc.collect()

# Cambiar el modelo a XGBoost
cls = make_pipeline(SimpleImputer(), XGBClassifier(max_depth=10, random_state=2345))  

# Realiza validación cruzada con k-fold (por ejemplo, k=5)
cv_scores = cross_val_score(cls, X_train, y_train, cv=6, scoring='roc_auc')

# Calcula el promedio de los puntajes de validación cruzada
mean_cv_score = cv_scores.mean()
print("Promedio de ROC AUC en validación cruzada:", mean_cv_score)

# Entrena el modelo final en todos los datos de entrenamiento
cls.fit(X_train, y_train)

# Calcula las predicciones en el conjunto de prueba
y_pred_test = cls.predict_proba(X_test)[:, cls.classes_ == 1].squeeze()

# Calcula el ROC AUC en el conjunto de prueba
roc_auc_test = roc_auc_score(y_test, y_pred_test)
print("ROC AUC en conjunto de prueba:", roc_auc_test)

# Predict on the evaluation set
eval_data = eval_data.drop(columns=["conversion"])
eval_data = eval_data.select_dtypes(include='number')
y_preds = cls.predict_proba(eval_data.drop(columns=["ROW_ID"]))[:, cls.classes_ == 1].squeeze()

# Make the submission file
submission_df = pd.DataFrame({"ROW_ID": eval_data["ROW_ID"], "conversion": y_preds})
submission_df["ROW_ID"] = submission_df["ROW_ID"].astype(int)
submission_df.to_csv("xgboost_model.csv", sep=",", index=False)