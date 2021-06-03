# MPI-ML

**MPI-ML** is parallel implementation of classification task.<br>
MPI-ML runs a few different classifiers on given dataset (loaded from csv file).<br>
Each classifier is parallely trained on the same dataset.

Testing dataset is equally divided and distributed to each process.<br>
Each process runs classification task on the received part of testing data.


Project was created to compare the performance and accuracy of different classifiers with the use of **Message Passing Interface** in Python.

## Dependencies
- MPI
- numpy
- pandas
- sklearn
- mpi4pi

## Build & Run

Install MPI (e.g. Ubuntu)
```
sudo apt install libmpich-dev
```

Install Python dependencies
```
sudo pip3 install sklearn pandas numpy mpi4py
```

Run
```
mpirun -n 4 python3 main.py
```

Note:<br>
**main.py** must be run by **mpirun** to make the execution parallel. Otherwise only one process will be created and as a result only one classifier will be run.<br><br>
Number of processes to be used for computation (4 in example) depends on number of classifiers you want to run parallely.<br>
Current version of contains four classifiers: **KNeighborsClassifier**, **DecisionTreeClassifier**, **MLPClassifier**, **SVC** therefore 4 processes were used for computation.<br>
If you want to run more classifiers parallely then you may want to use more processes - depending on your hardware.
