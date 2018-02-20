#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: rubenslopes


Estimações do projeto final da disciplina Topics in Nonparametric Statistics and Machine Learning course.
Professor: Raydonal Ospina Martínez

Base com 26.087 obs e 132 variáveis.

"""

import os
import numpy as np
import pandas as pd

#==============================================================================
# PREPARANDO A BASE
#==============================================================================

base_ml['ln_vav_def']=np.log(base_ml['vav_def'])
base_ml['ln_vav_def'].plot.hist()

#==============================================================================
# ESTIMAÇÕES
#==============================================================================

### OLS

'''
Var dependente: ln do valor avaliado
Regressores normais
'''
## Preparar base para estimação
base_mod1 = base_ml.drop('vav_def',axis=1).copy()
X = base_mod1.drop(['ln_vav_def'], axis=1)
import statsmodels.api as sm
X = sm.add_constant(X)
y = base_mod1['ln_vav_def']
## Estimar
model = sm.OLS(y,X).fit()
## Avaliar
ols_pred = model.predict(X)
model.summary()
print('R2:',model.rsquared)
print('R2 ajustado:',model.rsquared_adj)
print('RMSE:',np.sqrt(np.mean((y-ols_pred)**2)))

#==============================================================================

'''
A partir de agora, vou usar variáveis numéricas com variância padronizada conforme sugerido em classe e no livro Introduction to Statistical Learning. Faço isso para todos os regressores exceto para dummies.
'''
## Preparando base
base_mod2 = base_ml.drop('vav_def',axis=1).dropna().copy()
# Padronizar a variância
for i in ['QTDPAVEDSIMPLES','REACONSTRCOMUM','REACONSTRPRIVATIVA','REACONSTRTOTAL','READOLOTE','vitimas_cvli','roubo','drogas','estupros','area_pracas','pracas_parques','area_verde','areas_verdes','idade', 'cnefe_escolas','cnefe_saude', 'cnefe_bar_restaurante', 'cn_padaria_farmacia_mercado', 'numero', 'andar']:
    base_mod2[i] = base_mod2[i]/base_mod2[i].std()
    print('coluna',i, 'padronizada. Variância:',base_mod2[i].std())

#==============================================================================

### Post-Lasso

X = base_mod2.drop(['ln_vav_def'], axis=1)
y = base_mod2['ln_vav_def']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
from sklearn.linear_model import LassoCV
lassocv = LassoCV(cv=300, verbose=True, random_state=101, n_jobs=-1)
lassocv.fit(X_train,y_train)
selecao = pd.DataFrame(sorted(zip(np.around(lassocv.coef_,4), X.columns), reverse=True))
nova_selec = selecao[selecao[0]!=0]
nova_selec = list(nova_selec[1])
X_train = X_train[nova_selec]
X_train = sm.add_constant(X_train)
X_test = X_test[nova_selec]
X_test = sm.add_constant(X_test)
## Estimando OLS
model2 = sm.OLS(y_train,X_train).fit()
ols_pred = model2.predict(X_test)
model2.summary()
print('R2:',model2.rsquared)
print('RMSE',np.sqrt(np.mean((y_test-ols_pred)**2)))
y_postlasso_test = y_test

#==============================================================================

### Random Forests Regression

X = base_mod2.drop(['ln_vav_def'], axis=1)
y = base_mod2['ln_vav_def']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
## RODAR ALGO
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=100, random_state=101, n_jobs=-1, verbose=0)
rfr.fit(X_train, y_train)
rfr_pred = rfr.predict(X_test)
## AVALIAR
import sklearn.metrics
print(sklearn.metrics.r2_score(y_test, rfr_pred))
print('RMSE',np.sqrt(np.mean((y_test-rfr_pred)**2)))
sorted(zip(np.around(rfr.feature_importances_,4), X_train.columns), reverse=True)
y_rfr_test = y_test

#==============================================================================

### Gradient Boosting

X = base_mod2.drop(['ln_vav_def'], axis=1)
y = base_mod2['ln_vav_def']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
## RODAR ALGO
import xgboost as xgb
xgbalgo = xgb.XGBRegressor(seed=101, silent=False)
xgbalgo.fit(X_train, y_train)
## AVALIAR
xgb_pred = xgbalgo.predict(X_test)
print('R2:',sklearn.metrics.r2_score(y_test, xgb_pred))
print('MSE',np.sqrt(np.mean((y_test-xgb_pred)**2)))
y_boosting_test = y_test


#==============================================================================
# Standard Errors of RMSE by bootstrap
#==============================================================================

def std_RMSE_bs(y_test, preditos,n):
    '''Calcular desvio-padrão do RMSE por bootstrap.
    y_test: vetor do test_set da variável dependente
    preditos: vetor contendo os valores preditos pelo modelo.
    n: quantidade de subsamples do bootstrap. Ho (2016) usa 10.000.
    '''
    test_set = pd.merge(left=pd.DataFrame(y_test).reset_index(drop=True).reset_index(), right=pd.DataFrame(preditos).reset_index(drop=True).reset_index()).drop('index', axis=1).rename(columns={0:'preditos'}) # Criar novo DF para depois obter samples
    RMSE_distrib = np.array([]) # Nesse array vão entrar os 10 mil RMSE calculados por bootstrap. Essa é a distribuição do RMSE estimada por bs
    for i in range(n):
        subsample = test_set.sample(frac=0.5) # Ho faz com 50% da base, com reposição
        rmse = np.sqrt(np.mean((subsample['ln_vav_def']-subsample['preditos'])**2))
        print(i+1,'- RMSE:',rmse)
        RMSE_distrib = np.append(RMSE_distrib,rmse)
    print('Standard error of RMSE:',RMSE_distrib.std())
    return RMSE_distrib.std() # Finalmente obtemos o desvio-padrão da distribuição de RMSE para o test_set do modelo em questão

std_RMSE_lasso = std_RMSE_bs(y_postlasso_test, ols_pred,10000)
std_RMSE_rfr = std_RMSE_bs(y_rfr_test, rfr_pred,10000)
std_RMSE_xbg = std_RMSE_bs(y_boosting_test, xgb_pred,10000)


### --//-- ###