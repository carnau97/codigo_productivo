#!/usr/bin/env python
# coding: utf-8

# # ENTRENAMIENTO MODELO CHURN

## Librerías

#pip install pyyaml
#pip install pandas
#pip install scikit-learn
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle

import sys
sys.path.append('/Users/carmenarnau/Desktop/Aplicaciones_ML_202505/sesion2/codigo_productivo/churn_project_production/')
from utils.model_functions import predict_and_get_auc, chi_square



# Cargar el archivo YAML

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)
    
    
# # 1. Tablón entrenamiento

# ### Construcción dataset con variables input

# Cargamos datasets de diciembre

print("Cargamos datos")

fecha = '2023-12-01'

clientes_diciembre_df = pd.read_csv(config['data_paths']['clientes'] + fecha + '.csv', sep='|')
consumos_diciembre_df = pd.read_csv(config['data_paths']['consumos'] + fecha + '.csv', sep='|')
financiacion_diciembre_df = pd.read_csv(config['data_paths']['financiacion'] + fecha + '.csv',sep='|')
productos_diciembre_df = pd.read_csv(config['data_paths']['productos'] + fecha + '.csv', sep= '|')


# Unimos datasets de diciembre

df_diciembre = clientes_diciembre_df.merge(consumos_diciembre_df, on="id", how="left")
df_diciembre = df_diciembre.merge(financiacion_diciembre_df, on="id", how="left")
df_diciembre = df_diciembre.merge(productos_diciembre_df, on="id", how="left")


# ### Construcción de la columna target

# Cargamos clientes de enero

fecha_pred = '2024-01-01'
df_enero = pd.read_csv(config['data_paths']['clientes'] + fecha_pred + '.csv', sep= '|')

df_enero['target'] = 0
df_enero = df_enero[['id','target']]

# Hacemos left join de enero sobre diciembre, para que así aparezcan todos los clientes del dataset de diciembre
# Imputamos los NA de target con 1 (clientes que estaban en diciembre pero no en enero -> se han ido de la compañia)

df = pd.merge(df_diciembre, df_enero, on = 'id', how='left')
df = df.fillna({'target':1})

df = df.drop('id', axis = 1) # las columnas identificadoras no sirven para el modelo


# ### Corrección de inconsistencias

print("Limpieza de datos")

df2 = df.copy() # creamos una copia para no sobrescribir el df original


df2.loc[df2["vel_conexion"] < 0, 'vel_conexion'] = np.nan
df2.loc[df2['conexion']=='adsl', 'conexion'] = 'ADSL'


# ### Tratamiento de nulos

df2 = df2.fillna({'descuentos':'NO', 'financiacion':'NO', 'incidencia':'NO'})
df2 = df2.fillna({'num_dt':0, 'imp_financ':0})
df2 = df2.fillna({'vel_conexion':df2['vel_conexion'].mean()})


# ### Conversión de categóricas a numéricas

df2['financiacion'] = np.where(df2['financiacion']=="SI", 1, 0)
df2['incidencia'] = np.where(df2['incidencia']=="SI", 1, 0)
df2['descuentos'] = np.where(df2['descuentos']=="SI", 1, 0)


# ### Ingeniería de características

año_actual = datetime.now().year
df2['antiguedad'] = año_actual - df2['antiguedad']


# ### Selección previa

y = df2["target"]
X = df2.drop(columns=["target"])


var_significant, statistical_significance = chi_square(df2, 'target',  X.columns, 0.05)


X = X[var_significant]


# ### División en train y test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)


# ### Transformadores

# Obtenemos lista con los nombres de las columnas categóricas y numéricas por separado

cat_cols = X.select_dtypes(include = ['object']).columns
num_cols = X.select_dtypes(include = ['integer', 'float']).columns

# Definir transformadores para columnas categóricas y numéricas

print("Ajustamos pipeline")

transformadores = [
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop = "first"), 
     cat_cols),  # OneHotEncoder para las columnas categoricas
    ('scaler', StandardScaler(), num_cols)]  # StandardScaler para las columnas numéricas

# ColumnTransformer facilita aplicar diferentes transformaciones segun la columna
preprocesador = ColumnTransformer(transformadores)

# Crear el pipeline completo con OneHotEncoder seguido de StandardScaler
pipeline = Pipeline(steps=[
    ('preprocesador', preprocesador)])


# Ajustar el pipeline sobre el conjunto de datos de train y transformar los datos de train y test

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)


# # 2. Entrenamiento del modelo

print("Grid search")

params = {"max_depth": [5, 6, 7, 8],
          "min_samples_leaf": [10, 15, 20]}

rf = RandomForestClassifier()

rf_cv = GridSearchCV(rf, params, cv=3, scoring = 'roc_auc')

rf_cv.fit(X_train,y_train)

print("Mejores hiperparámetros: {}".format(str(rf_cv.best_params_)))

print("Entrenamos modelo final")
rf =  RandomForestClassifier(**rf_cv.best_params_, random_state = 0)
rf.fit(X_train, y_train)
predict_and_get_auc(rf, X_train, X_test, y_train, y_test)

# 3. Guardado

# Guardamos modelo

print("Guardamos objetos")

# [NEW]: PONEMOS FECHA DE EJECUCION EN EL NOMBRE DE LOS FICHEROS PARA TENER EL VERSIONADO

fecha_ejecucion = datetime.now().strftime("%Y%m%d")

pickle.dump(rf, open(config['model_paths']['trained_model'] + 'model_' + fecha_ejecucion + '.pkl', 'wb')) # guardar el modelo en formato pickle

# Guardamos pipeline

pickle.dump(pipeline, open(config['model_paths']['pipeline'] + 'pipeline_' + fecha_ejecucion + '.pkl', 'wb')) # guardar el modelo en formato pickle

# Guardamos variables input que han sido utilizadas

pd.DataFrame(var_significant, columns = ['variables']).to_csv(config['model_paths']['input_vars'] + 'var_significant_' + fecha_ejecucion + '.csv', index = False)

