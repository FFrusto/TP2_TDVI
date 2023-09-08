import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import os

# Lista de nombres de archivos de datos
data_files = ["/Users/franciscofrustoalvarado/Desktop/TD_VI/TP2_TDVI/competition_data.csv", "/Users/franciscofrustoalvarado/Desktop/TD_VI/TP2_TDVI/competition_data_1.csv", "/Users/franciscofrustoalvarado/Desktop/TD_VI/TP2_TDVI/competition_data_2.csv", "/Users/franciscofrustoalvarado/Desktop/TD_VI/TP2_TDVI/competition_data_3.csv", "/Users/franciscofrustoalvarado/Desktop/TD_VI/TP2_TDVI/competition_data_4.csv", "/Users/franciscofrustoalvarado/Desktop/TD_VI/TP2_TDVI/competition_data_5.csv", "/Users/franciscofrustoalvarado/Desktop/TD_VI/TP2_TDVI/competition_data_6.csv"]

for data_file in data_files:
    # Cargar el archivo de datos
    comp_data = pd.read_csv(data_file)
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
    cls.fit(X_train, y_train)

    # Predict on the evaluation set
    eval_data = eval_data.drop(columns=["conversion"])
    eval_data = eval_data.select_dtypes(include='number')
    y_preds = cls.predict_proba(eval_data.drop(columns=["ROW_ID"]))[:, cls.classes_ == 1].squeeze()

    # Make the submission file
    # submission_df = pd.DataFrame({"ROW_ID": eval_data["ROW_ID"], "conversion": y_preds})
    # submission_df["ROW_ID"] = submission_df["ROW_ID"].astype(int)
    # submission_df.to_csv(f"xgboost_model_{os.path.splitext(data_file)[0]}.csv", sep=",", index=False)

    print(f"Archivo: {data_file}")
    print(f"Score: {cls.score(X_train, y_train)}")
