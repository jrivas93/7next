import numpy as np
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import svm

def get_data():

    all_pokemon = []
    for i in range(1,699):
        res = requests.get(f"https://pokeapi.co/api/v2/pokemon/{i}")
        # print(res.text)
        # print(f"pokemon number {i}")
        all_pokemon.append(res)

    return all_pokemon



def process_dataset():

    df = pd.read_csv("pokemon.csv")

    # Removing columns that aren't present in the PokeAPI
    df = df.drop(df.iloc[:,1:19], axis=1)
    # print(df.head())

    # Removing additional columns that aren't in the PokeAPI
    df = df.drop(['base_egg_steps', 'base_happiness', 'base_total', 'capture_rate', 'classfication',
                  'experience_growth', 'japanese_name', 'percentage_male', 'pokedex_number',
                  'type2', 'generation', 'is_legendary', 'name'], axis=1)
    
    # Converting height units to match PokeAPI values
    df['height_m'] = df['height_m'] * 10
    df['weight_kg'] = df['weight_kg'] * 10

    # Creating new columns with the first two abilities for each Pokemon
    df['ability1'] = df['abilities'][0].split()[0].replace('[', '').replace(',', '').replace('\'', '')
    df['ability2'] = df['abilities'][0].split()[1].replace(']', '').replace(',', '').replace('\'', '')
    df = df.drop(['abilities'], axis=1)
    # print(df.head())

    # Dropping rows that have NaN values
    dropped_rows = df['height_m'].isna().sum()
    print(f"Number of dropped rows with NaN values: {dropped_rows}")
    df = df.dropna(axis=0)

    return df

def data_analysis(df):

    
    # Creating a plot to examine the type distribution of the Pokemon
    df['type1'].value_counts().plot(kind='barh')
    plt.title("Pokemon Type Distribution")
    plt.tight_layout()
    plt.savefig("pokemon_type_distribution.png")
    plt.clf()

    # Creating a correlation plot
    # Non numerical values are disregarded in the heatmap
    df_corr = df.corr(method='pearson')
    mask = np.zeros_like(df_corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(df_corr, vmax=1.0, vmin=-1.0, mask=mask, cmap="YlGnBu",  annot=True)

    plt.title("Feature Correlation")
    plt.tight_layout()
    plt.savefig("feature_heatmap.png")
    plt.clf
    

def clean_api_input(res):

    res = res.json()

    attack = res['stats'][1]['base_stat']
    defense = res['stats'][2]['base_stat']
    hp = res['stats'][0]['base_stat']
    sp_attack = res['stats'][3]['base_stat']
    sp_defense = res['stats'][4]['base_stat']
    speed = res['stats'][5]['base_stat']
    height = res['height']
    weight = res['weight']

    val_list = [[attack, defense, height, hp, sp_attack, sp_defense, speed, weight]]
    df = pd.DataFrame(val_list, columns =['attack', 'defense', 'height_m', 'hp', 'sp_attack', 'sp_defense', 'speed', 'weight_kg'], dtype = float)
    
    return df



if __name__ == "__main__":

    df = process_dataset()
    data_analysis(df)


    y = df['type1']
    X = df.drop(['type1', 'ability1', 'ability2'], axis=1)

    # Unique values for each type
    types = list(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = svm.SVC()
    clf.fit(X_train, y_train)   

    print("Cross validation scoring: ")
    print(cross_val_score(clf, X_train, y_train, cv=5))
    print("Cross validation scoring (F1 Macro): ")
    print(cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro'))
    
    print("Test score: ")
    print(clf.score(X_test, y_test))

    y_preds = clf.predict(X_test)
    report = classification_report(y_test, y_preds, target_names=types, output_dict=True)
    # print(report)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("classification_report.csv")

    
    
    

    #=============================================================================
    # Testing classifier from API response
    #=============================================================================
    res = requests.get(f"https://pokeapi.co/api/v2/pokemon/25")
    sample_input = clean_api_input(res)    
    type_pred = clf.predict(sample_input)[0]
    print(f"Predicted Pokemon type: {type_pred}")