# Importy

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# Wczytanie danych


data = load_breast_cancer()

print(data.keys())

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target


# Podstawowa analiza danych

print(df["target"].value_counts())
print(data.target_names)

# 0 - malignant (złośliwy)
# 1 - benign (łagodny)



# Przygotowanie danych


X = df.drop(columns=["target"])
y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Train:")
print(y_train.value_counts())

print("\nTest:")
print(y_test.value_counts())



# Logistic Regression


log_model = LogisticRegression(
    max_iter=10000
)

log_model.fit(
    X_train,
    y_train
)

y_log_pred = log_model.predict(X_test)


print("\n===== Logistic Regression =====")

print(f"Accuracy: {accuracy_score(y_test, y_log_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_log_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_log_pred):.3f}")
print(f"F1: {f1_score(y_test, y_log_pred):.3f}")

print(
    f"Confusion Matrix:\n{confusion_matrix(y_test, y_log_pred)}"
)


print("\nClassification report:")
print(classification_report(y_test, y_log_pred))


# Uwaga:
# W tym zbiorze sklearn:
# 0 = malignant (złośliwy)
# 1 = benign (łagodny)
# Dlatego analizując medycznie należy zwracać szczególną uwagę
# na recall dla klasy 0, ponieważ false negative oznacza przeoczenie raka.



# Decision Tree - ograniczona głębokość


tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)

y_tree_pred = tree_model.predict(X_test)


print("\n===== Decision Tree (max_depth=5) =====")

print(classification_report(y_test, y_tree_pred))


# Decision Tree - bez ograniczenia głębokości

tree_deep = DecisionTreeClassifier(
    max_depth=None,
    random_state=42
)

tree_deep.fit(
    X_train,
    y_train
)

y_tree_deep_pred = tree_deep.predict(X_test)


print("\n===== Deep Decision Tree =====")

print(classification_report(y_test, y_tree_deep_pred))


# Porównanie predict_proba()

print("\n===== Predict probabilities =====")

print("Logistic Regression:")
print(log_model.predict_proba(X_test.iloc[[0]]))

print("\nDecision Tree:")
print(tree_model.predict_proba(X_test.iloc[[0]]))


# Sprawdzenie overfittingu drzewa


print("\n===== Overfitting check =====")

print(
    f"Shallow train: {tree_model.score(X_train, y_train):.3f}"
)

print(
    f"Shallow test: {tree_model.score(X_test, y_test):.3f}"
)

print(
    f"Deep train: {tree_deep.score(X_train, y_train):.3f}"
)

print(
    f"Deep test: {tree_deep.score(X_test, y_test):.3f}"
)


# Wyniki pokazują lekki overfitting:
# głębokie drzewo osiąga 100% na treningu,
# ale gorzej radzi sobie na danych testowych.


# Wizualizacja drzewa


plt.figure(figsize=(20, 10))

tree.plot_tree(
    tree_model,
    feature_names=X.columns,
    class_names=["malignant", "benign"],
    filled=True
)

plt.show()



# Przykład najbardziej niepewnego przypadku


prob = log_model.predict_proba(X_test)

uncertain = np.abs(prob[:, 0] - 0.5).argmin()


print("\n===== Most uncertain example =====")

print("Logistic Regression:")
print(log_model.predict_proba(X_test.iloc[[uncertain]]))

print("\nDecision Tree:")
print(tree_model.predict_proba(X_test.iloc[[uncertain]]))


# Ciekawa różnica:
# Logistic Regression zwraca wynik bliżej 50/50,
# ponieważ wyznacza prawdopodobieństwo na podstawie funkcji modelu.
# Decision Tree może zwrócić bardzo wysoką pewność,
# ponieważ pacjent trafił do konkretnego liścia,
# gdzie w danych treningowych dominowała jedna klasa.

feature_importance_tree = pd.DataFrame({
    "feature": X.columns,
    "importance": tree_model.feature_importances_
})
#Waznosc kolejnych cech wedlug drzewka

feature_importance_tree = feature_importance_tree.sort_values(
    "importance",
    ascending=False
)

print(feature_importance_tree) #To samo ale posortowane
#Co ciekawe najistotniejsza cecha odpowiada za 71% waznosci wszystkich cech (worst radius)
#Jedna z przyczyn tego zjawisk ajest to, ze ta cecha jest silnie skorelowana z innymi
#*Nie chodzi o to ze odpowiada za 71% raka*
#Cechy z waznoscia 0 nie oznacza ze w ogole nie sa istotne
#Np przy log. reg. te cechy moglyby miec wieksza wage

#Skalujemy wagi
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

#Skalujemy w log
log_model_scaled = LogisticRegression(
    max_iter=10000
)

log_model_scaled.fit(
    X_train_scaled,
    y_train
)

y_log_scaled_pred = log_model_scaled.predict(X_test_scaled)
print(classification_report(y_test, y_log_scaled_pred))
print(classification_report(y_test, y_log_pred))
#Na tym zbiorze danych standaryzacja wag podniozla recall dla klasy 0 o 2 punkty procentowe

#Teraz porownujemy wagi dla drzewka i log
feature_importance_log = pd.DataFrame({
    "feature": X.columns,
    "importance": log_model_scaled.coef_.flatten()
})

feature_importance_log = feature_importance_log.sort_values(
    "importance",
    ascending=False
)
feature_importance_log["abs_importance"] = (
    feature_importance_log["importance"].abs()
)
feature_importance_log = feature_importance_log.sort_values(
    "abs_importance",
    ascending=False
)
print(feature_importance_log)
print(feature_importance_tree)




#Walidacja modelu

X_train, X_val, y_train, y_val = train_test_split(
    X_train, 
    y_train, 
    test_size=0.25,
    random_state=42,
    stratify=y_train
)
for depth in [2, 3, 4, 5, 6, None]:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)

    print(
        f"Depth: {depth} | "
        f"Train: {train_score:.3f} | "
        f"Validation: {val_score:.3f}"
    )
# 3 ostatnie modele maja taki sam poziom walidation, podejrzewamy ze najlepszym wyborem bedzie depht=5 (najmniejsze ryzyko overfittingu)
#Sprawdzamy dla depgt=5

X_train_final = pd.concat([X_train, X_val])
y_train_final = pd.concat([y_train, y_val])

final_tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

final_tree.fit(X_train_final, y_train_final)
final_test_score = final_tree.score(X_test, y_test)

print(f"Final test accuracy: {final_test_score:.3f}")
#Porownujac ten test z wczesniejszym deep/shallow widzimy, ze otrzymane wyniki sa prawie zgodne

tree_grid = DecisionTreeClassifier(
    random_state=42
)
param_grid = {
    "max_depth": [2, 3, 4, 5, 6, None]
}
grid_search = GridSearchCV(
    tree_grid,
    param_grid,
    cv=5,
    scoring="accuracy"
)
grid_search.fit(X_train_final, y_train_final)
print(grid_search.best_params_)
print(grid_search.best_score_)
#Znowu potwierdzenie, ze najlepiej sprawdza sie depht=5
#Sprawdzamy accuracy
best_tree = grid_search.best_estimator_

print(
    f"Test accuracy: "
    f"{best_tree.score(X_test, y_test):.3f}"
)


#Ostateczn eporownanie

y_log_final = log_model_scaled.predict(X_test_scaled)
y_tree_final = best_tree.predict(X_test)

results = {
    "Logistic Regression": [
        accuracy_score(y_test, y_log_final),
        precision_score(y_test, y_log_final),
        recall_score(y_test, y_log_final),
        f1_score(y_test, y_log_final)
    ],
    "Decision Tree": [
        accuracy_score(y_test, y_tree_final),
        precision_score(y_test, y_tree_final),
        recall_score(y_test, y_tree_final),
        f1_score(y_test, y_tree_final)
    ]
}

results_df = pd.DataFrame(
    results,
    index=["Accuracy", "Precision", "Recall", "F1"]
).T

print(results_df)
#Na tym zbiorze danych Logistic Regression osiagnelo lepsze wyniki, co istotne w typm przypadku: lepszy recall o 3 punkty procentowe
#Dadajemy jeszcze confusion matrix zeby miec besposredni wglad do recall

cm_log = confusion_matrix(y_test, y_log_final)
cm_tree = confusion_matrix(y_test, y_tree_final)

print("Logistic Regression:")
print(cm_log)

print("\nDecision Tree:")
print(cm_tree)
#Tutaj juz widzimy besposrednio, ze logistic regression ma przewage nad drzewkiem:
#dla log: przypada 41tn na 1fn
#dla tree: przypada 39tn na 6fn

#README
#Problem, ktorym sie zajmowalismy byla klasyfikacja gozow jako zlosliwe/lagodne. DataSet, na ktorych trenowalismy modele, to Breast Cancer Wisconsin.
#Porownywalismy ze soba dwa modele: Regresa Logistyczna vs Drzewko Decyzyjne. MOdele trenowalismy dzielac dane na zbiory treningowe i testowe.
#Pozniej optymalizowalismy hiperparametytry za pomoca validacji i GridSearchCV. Porownujac wyniki doszlismy do wniosku, ze najlepsza
#glebokosc drzewka, to max_dehpt=5. Zestawiajac ze soba oba, juz zoptymalizowane modele zauwazylismy roznice w najistotniejszych dla nas aspektych
#(w tym przypadku recall), na korzysc Logistic Regression. Upewnilismy sie zwracajac uwage na Confusion Matrix dla obu modeli (zwracajac uwage na wyniki TN i FN), co upewnilo nas w 
#tym przekonaniu. Ostatecznie Logistic Regression okazalo sie byc (na tym zbiorze danych) bardziej skutecznym modelem