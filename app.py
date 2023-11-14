from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


dataset = pd.read_csv("hemogramas.csv")
colunas = ["Eritrocitos","Hemoglobina","Hematocrito","HCM","VGM","CHGM","Metarrubrocitos","ProteonaPlasmatica","Leucocitos","Leucograma","Segmentados","Bastonetes","Blastos","Metamielocitos","Mielocitos","Linfocitos","Monocitos","Eosinofilos","Basofilos","Plaquetas"]
coluna_resultado = ["Diagnostico"]
dataset['Diagnostico'] = dataset['Diagnostico'].astype('category')

X_train, X_test, y_train, y_test = train_test_split(dataset[colunas], dataset[coluna_resultado], test_size=0.2, random_state=42)

#nome_da_coluna = "NomeDoenca"
#for indice, linha in dataset.iterrows():
#    valor_da_coluna = linha[nome_da_coluna]
#    print(f'Valor da coluna para a linha {indice}: {valor_da_coluna}')

def cria_modelo():
    return LinearRegression().fit(dataset[colunas], dataset[coluna_resultado])

modelo = cria_modelo()

disease_mapping = {
    1: "DRC (Doença Renal Cronica)",
    2: "Hipercolesterolemia",
    3: "Anemia",
    4: "Lesao Hepatica",
    5: "Infecçao Bacteriana",
    6: "Desidrataçao",
    7: "Infecçao Parasita",
    8: "Infecçao Viral",
    9: "DRC e Infeccao",
    10: "Neoplasia Hepatica",
    11: "Anemia Hemolitica",
    12: "Diabetes",
    13: "Pre-Diabetes",
    14: "Trombocitopenia e Inflamacao",
    15: "Anemia e Infecçao",
    16: "Anemia Hemoloide",
    17: "Pancreatite",
    18: "Inflamaçao Grave",
    19: "Hepatopatia",
    20: "Cardiotapia e DRC",
    21: "Hipoplasia Mieloide",
    }

app = Flask(__name__)

@app.route("/hemo", methods=["POST"])
def diagnostico():
    data = request.json
    Eritrocitos = data["eritrocitos"]
    Hemoglobina = data["hemoglobina"]
    Hematocrito = data["hematocrito"]
    HCM = data["hcm"]
    VGM = data["vgm"]
    CHGM = data["chgm"]
    Metarrubrocitos = data["metarrubrocitos"]
    ProteonaPlasmatica = data["proteonaPlasmatica"]
    Leucocitos = data["leucocitos"]
    Leucograma = data["leucograma"]
    Segmentados = data["segmentados"]
    Bastonetes = data["bastonetes"]
    Blastos = data["blastos"]
    Metamielocitos = data["metamielocitos"]
    Mielocitos = data["mielocitos"]
    Linfocitos = data["linfocitos"]
    Monocitos = data["monocitos"]
    Eosinofilos = data["eosinofilos"]
    Basofilos = data["basofilos"]
    Plaquetas = data["plaquetas"]
    
    print(Eritrocitos, Hemoglobina, Hematocrito, HCM, VGM, CHGM, Metarrubrocitos, ProteonaPlasmatica, Leucocitos, Leucograma, Segmentados, Bastonetes, Blastos, Metamielocitos, Mielocitos, Linfocitos, Monocitos, Eosinofilos, Basofilos, Plaquetas)

    #y_pred = modelo.predict(X_test)
    #print("Acurácia:", accuracy_score(y_test, y_pred))
    #print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))
    #print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

    dados_consulta = np.array([[Eritrocitos, Hemoglobina, Hematocrito, HCM, VGM, CHGM, Metarrubrocitos, ProteonaPlasmatica, Leucocitos, Leucograma, Segmentados, Bastonetes, Blastos, Metamielocitos, Mielocitos, Linfocitos, Monocitos, Eosinofilos, Basofilos, Plaquetas]])
    predicao = modelo.predict(dados_consulta)

    nomedoenca_numerico = int(predicao[0])

    if nomedoenca_numerico in disease_mapping:
        nomedoenca = disease_mapping[nomedoenca_numerico]
    else:
        nomedoenca = "Unknown Disease"

    return jsonify({"mensagem": "Diagnostico: {}".format(nomedoenca)})

if __name__ == "__main__":
    app.run(debug=True)