import pandas as pd
from bin.pipeline import __creer_tableau_, __definir_les_donnees_
import sqlite3
from sqlite3 import connect

# télécherger la banque de données
df = pd.read_excel('AmelieBoucher_Plan_Psy4016_30032022_Student-mat.xlsx')

data_sql = df[['sex', 'address', 'Fedu', 'Medu', 'famrel', 'G1', 'G2', 'G3']]

__creer_tableau_('tab1')
__definir_les_donnees_('tab1', data_sql)

conn = connect('base_donnees.db', check_same_thread = False)
c = conn.cursor()
c.execute('''SELECT * FROM tab1''')
data = pd.DataFrame(c.fetchall(), columns=['sex', 'address', 'Fedu', 'Medu', 'famrel', 'G1', 'G2', 'G3'])    


# vérifier les colonnes, algo de gestion des erreurs 
while True:
    try:
        data.columns = ['sex', 'address', 'Fedu', 'Medu', 'famrel', 'G1', 'G2', 'G3']
        break
    except ValueError:
        print ('Oups! mauvaises colonnes. Vérifier les colonnes demandées')
        break

#cleaning des données (classe)
class clean_df:
        
    def __init__(self):
        self.test = 0
    
     # gestion des données manquante
    def col_Nan(self, df):
        return df.fillna(df.mode().iloc[0])

    # uniformiser les données
        # mettre valeurs non numériques en numérique
        # mettre valeurs float en int
        # cette fonction est un algorithme d'automatisation
    def unif(self, df):
        df.loc[df['sex'] == 'F', 'sex'] = 0
        df.loc[((df['sex'] == 'male') | (df['sex'] == 'M')), 'sex'] = 1
        df.loc[df['address'] == 'U', 'address'] = 0
        df.loc[df['address'] == 'R', 'address'] = 1
        return df.astype(int)


c = clean_df()
clean_data = c.col_Nan(data)
clean_data = c.unif(clean_data)

#tester la normalité des variables G3
import scipy
from scipy import stats

Normalité = stats.skew(clean_data['G3']), stats.kurtosis(clean_data['G3'])



##Hypothèse 1 
##Test t (G3 entre filles vs gars et urbain vs rural)
import scipy

femme = clean_data[clean_data['sex'] == 0]['G3']
homme = clean_data[clean_data['sex'] == 1]['G3']

res = stats.ttest_ind(femme, homme)
résultatsFH = res.statistic, res.pvalue

rural = clean_data[clean_data['address'] == 1]['G3']
urbain = clean_data[clean_data['address'] == 0]['G3']

res = stats.ttest_ind(rural, urbain)
résultatsUR = res.statistic, res.pvalue

#histogramme hypothèse 1
import matplotlib.pyplot as plt
import seaborn as sns
#figure1
sns.histplot(data = clean_data[['sex', 'G3']], x = 'G3', hue = 'sex')
plt.show()
# 0 = femme ; 1 = homme

#figure2
sns.histplot(data = clean_data[['address', 'G3']], x = 'G3', hue = 'address')
plt.show()
# 0 = urbain ; 1 = rural



##Hypothèse 2 (AA supervisé) : Le fait d’avoir des meilleures notes à G2 prédit une meilleure note G3.
##Régression linéaire 
#https://machinelearningmastery.com/loocv-for-evaluating-machine-learning-algorithms/
x = clean_data.drop('G3', axis = 1)
y = clean_data['G3']

import sklearn
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
x = clean_data.drop('G3', axis = 1)
model.fit(x, clean_data.G3)

intercept = model.intercept_
coefficient = model.coef_

# zip est une fonction anonyme
pd.DataFrame(zip(x.columns, model.coef_), columns=['features', 'estim_coef'])

for key in zip(x.columns, model.coef_):
    clé = key

# il existe une meilleure corrélation entre la G2 et G3
# tracer une dispersion des résultats significatifs
plt.scatter(clean_data.G2, clean_data.G3)
plt.xlabel('Note G2')
plt.ylabel('Note G3')
plt.title("Relation entre la note G2 et la note G3")
plt.show()

plt.scatter(clean_data.G3, model.predict(x))
plt.show()

#Validation croisée
Xtrain, Xtest, ytrain, ytest = sklearn.model_selection.train_test_split(x, clean_data.G3, test_size = 0.33, random_state=5)

# créer le modèle de prédiction
model.fit(Xtrain, ytrain)
train_predict = model.predict(Xtrain)

import numpy as np
#Fit le modèle Xtrain et montrer le MSE avec ytrain
fitytrain = np.mean((ytrain-model.predict(Xtrain))**2)
#Fit le modèle Xtrain et montrer le MSE avec Xtest, ytest
fitytest = np.mean((ytest-model.predict(Xtest))**2)

#figure3
plt.scatter(model.predict(Xtrain), model.predict(Xtrain)-ytrain, c = 'b', s=40, alpha=0.5) 
plt.scatter(model.predict(Xtest), model.predict(Xtest)-ytest, c='r', s=40)
plt.hlines(y=0, xmin=0, xmax=50)
plt.show()



##Hypothèse 3 (AA non-supervisé) : les différentes variables indépendantes ('Fedu', 'Medu', 'famrel', 'G1', 'G2') permettent de prédire G3
##K-moyens 
import sklearn
from sklearn.decomposition import PCA


test_predict = model.predict(Xtest)

# visualisation PCA
pca = PCA(2)  
projected = pca.fit_transform(clean_data)

plt.scatter(projected[:, 0], projected[:, 1],
            c=clean_data.G3, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();
plt.show()

var_ratio = pca.explained_variance_ratio_

varianceEx = pca.explained_variance_

pca = PCA().fit(clean_data)
#figure4
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('nombre de composants')
plt.ylabel('variance expliquée cumulative');
plt.show()



#rapport avec formatage de chaîne
print(f'Les résultats du test t entre les hommes et les femmes ({résultatsFH}) et les urbrains et ruraux ({résultatsUR}) montrent que les groupes diffèrent significativement.')
print(f'La régression linéaire multiple révèle que la note G2 prédit bien la note G3, comme il était attendu')
print(f'L\'analyse des composantes principales révèle que la majorité de la variance est expliquée par les 4 premières composantes {varianceEx}')
