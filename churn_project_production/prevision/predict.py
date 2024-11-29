#!/usr/bin/env python
# coding: utf-8

# # PREVISIÓN MODELO CHURN


# ## Librerías

import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import argparse

# Cargamos parametro

parser = argparse.ArgumentParser()

# Definimos el argumento model_version
parser.add_argument("--model_version", type=str, required=True, help="Versión del modelo a usar")

# Parseamos los argumentos
args = parser.parse_args()
model_version = args.model_version

print(f"La versión del modelo es: {model_version}")

# Cargar el archivo YAML

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)
    
# ## Carga de modelos, variables y transformaciones

pipeline = pickle.load(open(config['model_paths']['pipeline'] + 'pipeline_' + model_version + '.pkl', 'rb')) # scaler
modelo = pickle.load(open(config['model_paths']['trained_model'] + 'model_' + model_version + '.pkl', 'rb')) # modelo entrenado
var_significant = pd.read_csv(config['model_paths']['input_vars'] + 'var_significant_' + model_version + '.csv') # variables a utilizar


# # 1. Tablón de previsión

# Cargamos variables input a fecha actual

fecha = '2024-01-01'

clientes_df = pd.read_csv(config['data_paths']['clientes'] + fecha + '.csv', sep='|')
consumos_df = pd.read_csv(config['data_paths']['consumos'] + fecha + '.csv', sep='|')
financiacion_df = pd.read_csv(config['data_paths']['financiacion'] + fecha + '.csv',sep='|')
productos_df = pd.read_csv(config['data_paths']['productos'] + fecha + '.csv', sep= '|')

# Unimos variables input en un dataframe

df = clientes_df.merge(consumos_df, on="id", how="left")
df = df.merge(financiacion_df, on="id", how="left")
df = df.merge(productos_df, on="id", how="left")


# Se realiza exactamente el mismo preprocesamiento que el realizado sobre el tablon de train:


df2 = df.copy() # creamos una copia para no sobrescribir el df original

df2.loc[df2["vel_conexion"] < 0, 'vel_conexion'] = np.nan
df2.loc[df2['conexion']=='adsl', 'conexion'] = 'ADSL'


df2 = df2.fillna({'descuentos':'NO', 'financiacion':'NO', 'incidencia':'NO'})
df2 = df2.fillna({'num_dt':0, 'imp_financ':0})
df2 = df2.fillna({'vel_conexion':df2['vel_conexion'].mean()})


df2['financiacion'] = np.where(df2['financiacion']=="SI", 1, 0)
df2['incidencia'] = np.where(df2['incidencia']=="SI", 1, 0)
df2['descuentos'] = np.where(df2['descuentos']=="SI", 1, 0)


año_actual = datetime.now().year

df2['antiguedad'] = año_actual - df2['antiguedad']


# # 2. Selección variables y transformadores


# Seleccionamos variables cargadas

variables = var_significant['variables'].tolist()

X = df2[variables]


# Aplicamos transformador cargado

X_fin = pipeline.transform(X)


# # 3. Obtenemos predicciones


# Aplicamos modelo cargado

predicciones = modelo.predict_proba(X_fin)


# Debemos tener la prediccion asociada a cada idcliente:

df['probabilidad'] = predicciones[:,1]
df_pred = df[['id', 'probabilidad']]

df_pred.loc[:,'decil'] = pd.qcut(df_pred.loc[:,'probabilidad'], 10, labels = range(1,11))

fecha_ejecucion = datetime.now().strftime("%Y%m%d")

df_pred.to_csv(config['prediction_paths']['predictions'] + 'predicciones_' + fecha_ejecucion + '.csv', index = False)

