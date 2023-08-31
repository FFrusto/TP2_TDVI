import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Load the competition data
comp_data = pd.read_csv("competition_data.csv")

# Split into training and evaluation samples
train_data = comp_data[comp_data["ROW_ID"].isna()]
eval_data = comp_data[comp_data["ROW_ID"].notna()]
del comp_data
gc.collect()

# Dividir los datos en conjunto de entrenamiento y prueba (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    train_data.drop(columns=["conversion", "ROW_ID"]).select_dtypes(include='number'),
    train_data["conversion"],
    test_size=0.1,  # Proporción para el conjunto de prueba
    random_state=42  # Semilla aleatoria para reproducibilidad
)
del train_data
gc.collect()
# Create pipeline with SimpleImputer and DecisionTreeClassifier
cls = make_pipeline(SimpleImputer(), DecisionTreeClassifier(max_depth=8, random_state=2345))

# Perform cross-validation
cv_scores = cross_val_score(cls, X_train, y_train, cv=5, scoring='accuracy')  # Change scoring as needed

print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# Train the model on the entire training set
cls.fit(X_train, y_train)

# Predict on the evaluation set
eval_data = eval_data.drop(columns=["conversion"])
eval_data = eval_data.select_dtypes(include='number')
y_preds = cls.predict_proba(eval_data.drop(columns=["ROW_ID"]))[:, cls.classes_ == 1].squeeze()

# Make the submission file
submission_df = pd.DataFrame({"ROW_ID": eval_data["ROW_ID"], "conversion": y_preds})
submission_df["ROW_ID"] = submission_df["ROW_ID"].astype(int)
submission_df.to_csv("basic_model_with_cv.csv", sep=",", index=False)

print("Final Model Score:", cls.score(X_test, y_test))
