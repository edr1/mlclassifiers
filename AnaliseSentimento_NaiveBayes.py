import logging
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
import nltk
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Configure Logging
#logging.basicConfig(filename="NaiveBayes_AnaliseSentimento.log", format='%(asctime)s %(message)s', filemode='w') 
#logger=logging.getLogger() 
#logger.setLevel(logging.INFO)
nltk.download('stopwords')
PTBRStopsWords = set(stopwords.words("portuguese"))

def prepare_data ():
    # Read CSV File
    dataset = pd.read_csv('./AVALIACOES_ALUNOS.csv',encoding='utf-8')
    # Get DataSets for Feature and Label
    descricoes = dataset["DESCRICAO"].values
    categorias = dataset["CATEGORIA"].values
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(descricoes, categorias, train_size=0.6)
    print("\n\tTRAINING DATASET: Texto-train = {}  Categorias-train = {}".format(len(X_train), len(Y_train)))
    print("\tTESTING  DATASET: Texto-test  = {}  Categorias-test  = {}".format(len(X_test), len(Y_test)))
    return X_train, X_test, Y_train, Y_test

def process_model (X_train, X_test, Y_train, Y_test):
    # Bag of Words e o algoritmo Naive Bayes Multinomial
    #vectorizer = CountVectorizer(analyzer = "word", strip_accents = 'unicode', tokenizer=None, preprocessor=None, stop_words=PTBRStopsWords)
    #vectorizer = CountVectorizer(analyzer = "word", strip_accents = 'unicode', stop_words=PTBRStopsWords)
    vectorizer = CountVectorizer(analyzer = "word", strip_accents = 'unicode', stop_words=None)
    naivBayes = MultinomialNB()
    pipeline = Pipeline([('vect', vectorizer), ('naivBayes', naivBayes)])
    pipeline.fit(X_train, Y_train)
    pipeline_score = pipeline.score(X_test, Y_test)
    print("\n-----------------------------------------------")
    print("\nPipeline_final_Score: {}".format(pipeline_score))
    print("\n-----------------------------------------------")
    return pipeline, naivBayes, vectorizer

def predict_data(pipeline):
    X_test_1 = ["Temos computadores e internet disponíveis sempre que precisamos, biblioteca ampla, ótima estrutura, todos os funcionários excelentes, sem exceção de nenhum, ótima segurança e etc Indico sempre a faculdade",
          "Secretaria um pouco desorganizadas e confusa, acho o sistema de notas um pouco injusto, o jeito que os professores avaliam e etc.",
          "A universidade tem salas bagunçadas", "Boa estrutura na sala de aulas e bons professores"]
    #X_test_pred_1 = vectorizer.transform(X_test_1)
    X_test_pred_1 = pipeline.predict(X_test_1)
    print("\tX_test_1:      {}".format(X_test_1))
    print("\tX_test_pred_1: {}".format(X_test_pred_1))

def evaluate_model(model, vectorizer, X_train, X_test, Y_test):
    # Validação cruzada
    X_test_transf = vectorizer.fit_transform(X_test)
    Y_test_pred = cross_val_predict(model, X_test_transf, Y_test)
    accuracy_score = metrics.accuracy_score(Y_test, Y_test_pred)
    print("\n-----------------------------------------------")
    print("\nMultinomial Naive Bayes - Accuracy_Score: {}".format(accuracy_score))
    print("\n-----------------------------------------------")
    # Relatorio de Classificação e Matriz de Confusão
    #    : precision = true positive / (true positive + false positive)
    #    : recall    = true positive / (true positive + false negative)
    #    : f1-score  = 2 * ((precision * recall) / (precision + recall))
    labels = ["Positivo", "Negativo", "Neutro"]
    classificationReport = metrics.classification_report(Y_test, Y_test_pred, labels)
    print("\n {}".format(classificationReport))
    plot_classification_report(classificationReport)
    confusionMatrix = confusion_matrix(Y_test, Y_test_pred, labels)
    plot_confusion_matrix(confusionMatrix, labels)


def plot_confusion_matrix(mat, labels):
    plt.clf()
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=labels, yticklabels=labels)
    plt.title("confusion_matrix")
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.show()

def plot_classification_report(cr, with_avg_total=False, cmap=plt.cm.Blues):
    lines = cr.split('\n')
    classes = []
    plotMat = []
    for line in lines[2 : (len(lines) - 3)]:
        t = line.split()
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        plotMat.append(v)
    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)
    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title('Classification report')
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    plt.show()

def main():
    X_train, X_test, Y_train, Y_test = prepare_data()
    pipeline, model, vectorizer = process_model(X_train, X_test, Y_train, Y_test)
    predict_data(pipeline)
    evaluate_model(model, vectorizer, X_train, X_test, Y_test)

if __name__ == "__main__":
    main()