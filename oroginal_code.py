# %%
# Import all the necessary libraries 

# %%
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# %%
# Load dataset
df = pd.read_csv(r"C:\Users\shehz\OneDrive\Desktop\deployment2\cleaned_laptops.csv") 

# %%
# Features & Target
X = df.drop("Price", axis=1)
y = df["Price"]

# %%
categorical_cols = ['Discount', 'Brands', 'processor']
numerical_cols = [
     'ratings', 'Original_price', 
    'number_of_ratings', 'number_of_reviews', 'RAM', 'Storage'
]

# %%
# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ]
)


# %%
# Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', LinearRegression())
])

# %%
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
model_pipeline.fit(X_train, y_train)

# %%
# Save model
joblib.dump(model_pipeline, "model.pkl")

# %%
import sklearn
sklearn.__version__


# %%


# %%



