"""
e-Residency application fraud screen -- CORRECTED evaluation pipeline.

Same model as project_pipeline.py (StandardScaler + 1-NN on the six numeric features).
What changed:
  * The evaluation now splits by APPLICANT (date_of_birth + citizenship), not by row,
    so no person's resubmission can sit in the training data while another of their
    applications is being scored.  <- the real fix
  * The scaler lives inside a Pipeline, so it is refit on the training rows of every fold
    (hygiene; does not change the score).
Identity fields are still never used as model features -- only to define the folds.
"""

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("applications.csv")

X = df.drop(columns=["submission_id", "date_of_birth", "citizenship", "label"])
y = df["label"]
applicant = df["date_of_birth"] + "|" + df["citizenship"]   # grouping key, not a feature

model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))

# Leave-one-APPLICANT-out: the grouped equivalent of LeaveOneOut()
pred = cross_val_predict(model, X, y, cv=LeaveOneGroupOut(), groups=applicant)
acc = (pred == y).mean()
print(f"True accuracy (leave-one-applicant-out): {acc:.4f}")
print(f"FLAG: SIGMOID_{round(acc * 100)}")
