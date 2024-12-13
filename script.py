import pandas
import pyarrow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

TITANIC: pandas.DataFrame = pandas.read_csv(
    "./titanic.csv",
    dtype_backend="pyarrow",
)

## Author: Gabriel Hannonen Vieira

# Bloco limpando os dados e convertendo o parametro Sex para 0 e 1
sex = ["male", "female"]
enc = OrdinalEncoder(categories=[sex])
titanic_limpo = TITANIC
titanic_limpo["Sex"] = enc.fit_transform(titanic_limpo[["Sex"]])
titanic_limpo = titanic_limpo.drop(columns=["Name", "Ticket", "Cabin", "Embarked"])


## bloco progressão linear
def progressaolinear(titanic):
    X = titanic.iloc[  ##parametros utilizados para treino (Todos menos Name, Age,Ticket,Cabin e Embarked)
        :, [True, False, True, True, False, True, True, True]
    ]
    y = titanic.iloc[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=1
    )  ## separação em grupos de teste e treino
    # print(X_train)
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)
    predicao = modelo.predict(X)
    print(modelo.score(X, y))  # verificação do score do modelo
    # print(len(predicao)) ## teste para ver se todos os dados estavam sendo testados
    return predicao


## Bloco K-Nearest Neighbors
def knn(titanic):
    X = titanic.iloc[  ##parametros utilizados para treino (Todos menos Name, Age,Ticket,Cabin e Embarked)
        :, [True, False, True, True, False, True, True, True]
    ]
    y = titanic.iloc[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=1
    )  ## separação em grupos de teste e treino
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(X_train, y_train)
    predicao = knn.predict(X)
    # print(len(predicao)) ## teste para ver se todos os dados estavam sendo testados
    print(knn.score(X, y))  # verificação do score do modelo
    return predicao


## Arvore de decisão
def arvore_decisao(titanic):
    X = titanic.iloc[  ##parametros utilizados para treino (Todos menos Name, Age,Ticket,Cabin e Embarked)
        :, [True, False, True, True, False, True, True, True]
    ]
    y = titanic.iloc[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=13
    )  ## separação em grupos de teste e treino
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    predicao = dtc.predict(X)
    # print(len(predicao)) ## teste para ver se a lengh correspondia
    print(dtc.score(X, y))  # verificação do score do modelo
    return predicao


# Função para a construção do CSV
def escrever_csv(titanic, predicao, nome_modelo):
    df_final = titanic
    df_final = df_final.reindex(  # reformatando o df
        columns=[
            "PassengerId",
            "Pclass",
            "Name",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Ticket",
            "Fare",
            "Cabin",
            "Embarked",
            "Survived",
        ]
    )
    df_final.insert(0, "Modelo", [nome_modelo for i in range(len(predicao))])
    df_final.insert(12, "Predicted", predicao)
    return df_final


# Rodando os modelos
PlResults = progressaolinear(titanic_limpo)
KnnResults = knn(titanic_limpo)
TreeResults = arvore_decisao(titanic_limpo)


lista_modelos = [
    ("progressao_linear", PlResults),
    ("knn", KnnResults),
    ("arvore_decisao", TreeResults),
]
df_resultados = []

for modelo in lista_modelos:
    df_resultados.append(escrever_csv(TITANIC, modelo[1], modelo[0]))

# salvandos os modelos no CSV
df_resultados_finais = pandas.concat(df_resultados)
df_resultados_finais.to_csv("titanic_treinado.csv", sep=";", decimal=",")
