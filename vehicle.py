from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from scipy.stats import randint
import joblib

try:
    df = pd.read_csv('dataset.csv')
except FileNotFoundError:
    print("Error : File not Found")

# print(df.head())

df = df[(df['price'] > 1000) & (df['price'] < 200000)]
df = df[df['mileage'] >= 0]

num_cols = ['year','cylinders','mileage','doors']
cat_cols = ['name','make','model','engine','fuel','transmission','trim','body','exterior_color', 'interior_color', 'drivetrain']

num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))
])

x = df[num_cols+cat_cols]
y = df['price']

col_transform = ColumnTransformer(
    transformers=[
        ('num_pipeline', num_pipeline,num_cols),
        ('cat_pipeline', cat_pipeline,cat_cols)
    ],remainder='drop',n_jobs=-1)

rfr = RandomForestRegressor()

full_pipeline = Pipeline(steps=[
    ('preprocess', col_transform),
    ('regressor', RandomForestRegressor(random_state=42))
])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

param_dict = {
    'regressor__n_estimators': randint(50, 300),
    'regressor__max_depth': randint(3, 20),
    'regressor__min_samples_split': randint(2, 10),
    'regressor__min_samples_leaf': randint(1, 10),
    'regressor__max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    full_pipeline,
    param_distributions=param_dict,
    n_iter=20,
    cv=5,
    n_jobs=-1,
    random_state=42,
    scoring='neg_mean_squared_error'
)

random_search.fit(x_train,y_train)
# print('Best parameter:',random_search.best_params_)
best_params = random_search.best_params_

final_model = Pipeline(steps=[
    ('preprocessor',col_transform),
    ('regressor',RandomForestRegressor(
        n_estimators=best_params['regressor__n_estimators'],
        max_depth=best_params['regressor__max_depth'],
        min_samples_split=best_params['regressor__min_samples_split'],
        min_samples_leaf=best_params['regressor__min_samples_leaf'],
        max_features=best_params['regressor__max_features'],
        random_state=42
    ))
])

final_model.fit(x_train,y_train)
joblib.dump(final_model, 'vehicle_price_model.pkl')
print("Model saved successfully!")

y_pred = final_model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print('mean squared error: ',mse)
print('mean absolute error: ',mae)
print('R2: ',r2)
print('Model score: ',final_model.fit(x_train,y_train).score(x_test,y_test))