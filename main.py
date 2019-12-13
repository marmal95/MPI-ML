import numpy
import pandas
from mpi4py import MPI
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

MASTER_RANK = 0
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def log(text):
    print('[' + str(rank) + ']: ' + text)


def get_classifier():
    switcher = {
        0: KNeighborsClassifier(n_neighbors=1, algorithm='brute'),
        1: DecisionTreeClassifier(),
        2: MLPClassifier(),
        3: SVC()
    }
    return switcher.get(rank)


def distribute_data():
    train_data = train_labels = split_test_data = split_test_labels = []

    if rank == MASTER_RANK:
        data = pandas.read_csv('smartphone_activity_dataset.csv', delimiter=',')
        data_labels = data[data.columns[[-1]]]
        data_features = data.drop(data.columns[[-1]], axis=1)

        scaler = MinMaxScaler()
        scaler.fit(data_features)
        data_features = scaler.transform(data_features)

        (train_data, test_data, train_labels, test_labels) = train_test_split(data_features, data_labels, test_size=0.1)
        log('Train data length: ' + str(len(train_data)))

        split_test_data = numpy.array_split(test_data, size)
        split_test_labels = numpy.array_split(test_labels, size)

    comm.Barrier()
    train_data = comm.bcast(train_data, root=MASTER_RANK)
    train_labels = comm.bcast(train_labels, root=MASTER_RANK)
    proc_test_data = comm.scatter(split_test_data, root=MASTER_RANK)
    proc_test_labels = comm.scatter(split_test_labels, root=MASTER_RANK)

    return train_data, proc_test_data, train_labels, proc_test_labels


def train(train_data, train_labels):
    classifier = get_classifier()
    classifier.fit(train_data, train_labels.values.ravel())
    return classifier


def predict(classifier, proc_test_data, proc_test_labels):
    proc_predicted = classifier.predict(proc_test_data)
    proc_score = metrics.accuracy_score(proc_test_labels, proc_predicted)
    log(str(type(classifier).__name__) + ': ' + str(proc_score))


def main():
    train_data, proc_test_data, train_labels, proc_test_labels = distribute_data()
    classifier = train(train_data, train_labels)
    predict(classifier, proc_test_data, proc_test_labels)


if __name__ == '__main__':
    main()
