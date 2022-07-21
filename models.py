from sklearn import tree, ensemble, linear_model, cluster, neighbors, preprocessing, model_selection
import sklearn
import pandas as pd
import numpy as np
import graphviz
import matplotlib.pyplot as plt
from collections import Counter
import xgboost as xgb
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

cand_dict = {
    0: 'sandersb',
    1: 'klobuchara',
    2: 'buttigiegp',
    3: 'yanga',
    4: 'bidenj',
    5: 'warrene',
    6: 'steyert',
    7: 'bennetm',
    8: 'bloombergm',
    9: 'delaneyj',
    10: 'gabbardt',
    11: 'patrickd',
    12: 'uncommitted',
    13: 'other'
}


def get_winners(results_df):
    results = results_df.drop(columns = ['fips', 'name', 'precincts', 'votes', 'reporting'])

    results = results.to_numpy()
    winners = np.nanargmax(results, axis = 1)

    return winners

def rep_dem(url):
    res = pd.read_json(url)
    res['fips'] = res['fipscode']
    res = res[res['raceid'] == 17076]

    #race_df = res[res['winner']]
    race_df = res.sort_values('votepct', ascending=False).drop_duplicates(['fips'])

    race_df = race_df.sort_values('fips')

    le = preprocessing.LabelEncoder()
    winners = le.fit_transform(race_df['party'].to_numpy())

    return winners


def preprocess(train_x_file, train_y_file, test_x_file, test_y_file):
    train_inputs = pd.read_csv(train_x_file, index_col = 0).to_numpy()

    train_y = pd.read_csv(train_y_file, index_col = 0)
    train_labels = train_y['winner'].to_numpy()

    test_inputs = pd.read_csv(test_x_file, index_col = 0).to_numpy()

    test_y = pd.read_csv(test_y_file, index_col = 0)
    test_labels = test_y['winner'].to_numpy()


    #shuffle training
    indices = np.arange(len(train_labels))
    np.random.shuffle(indices)

    train_inputs = train_inputs[indices, :]
    train_labels = train_labels[indices]


    return train_inputs, train_labels, test_inputs, test_labels




iowa_x = 'data/iowa_inputs.csv'
iowa_y = 'data/iowa_results.csv'
iowa_z = 'data/iowa_dem.csv'

def iowa_data(gov = False):
    iowa_inp = pd.read_csv(iowa_x, index_col = 0)
    iowa_labels = pd.read_csv(iowa_y, index_col = 0)

    iowa_dem = pd.read_csv(iowa_z)

    iowa_inp = iowa_inp.merge(iowa_dem, on='fips', how = 'left')

    #print(iowa_inp)
    if gov:
        labels = rep_dem('https://www.politico.com/election-results/2018/iowa/county.json')
    else:
        labels = get_winners(iowa_labels)

    iowa_inputs_pop = iowa_inp['B01001_001E'].to_numpy()

    iowa_inp = iowa_inp.drop(columns = ['fips', 'B01001_001E']).to_numpy()

    iowa_inp = np.transpose(np.transpose(iowa_inp) / iowa_inputs_pop)

    indices = np.arange(len(iowa_labels))
    np.random.shuffle(indices)

    iowa_inp = iowa_inp[indices, :]
    labels = labels[indices]

    train = round(len(iowa_labels) * 0.8)

    iowa_train_x = iowa_inp[:train, :]
    iowa_train_y = labels[:train]

    iowa_test_x = iowa_inp[train:, :]
    iowa_test_y = labels[train:]

    return iowa_train_x, iowa_train_y, iowa_test_x, iowa_test_y



def train(train_x, train_y, test_x, test_y, classification = True):

    if classification:
        tree_fit = tree.DecisionTreeClassifier(random_state=0)
        train_fit = tree_fit.fit(train_x, train_y)

        accuracy = tree_fit.score(test_x, test_y)

        dot_data = tree.export_graphviz(tree_fit, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render("data/tree")
        train_score = tree_fit.score(train_x, train_y)
        print('======================tree=========================')
        print('test:', accuracy)
        print('train:', train_score)


        rf = ensemble.RandomForestClassifier(min_samples_leaf = 3, random_state=0)
        train_fit = rf.fit(train_x, train_y)
        accuracy = rf.score(test_x, test_y)
        train_score = rf.score(train_x, train_y)
        print('==================random forest====================')
        print('test:', accuracy)
        print('train:', train_score)
        print(rf.feature_importances_)


        xgb_train = xgb.DMatrix(train_x, label = train_y)
        xgb_test = xgb.DMatrix(test_x)
        xgb_tree = xgb.train({'objective': 'multi:softmax', 'num_class':14, 'gamma': 5}, xgb_train)

        predictions = xgb_tree.predict(xgb_test)
        correct = np.equal(predictions, test_y).astype(int)
        accuracy = np.sum(correct) / len(test_y)

        train_pred = xgb_tree.predict(xgb_train)
        correct = np.equal(train_pred, train_y).astype(int)
        train_score = np.sum(correct) / len(train_y)

        print('======================xgb==========================')
        print('test:', accuracy)
        print('train:', train_score)


        svm = sklearn.svm.SVC()
        train_fit = svm.fit(train_x, train_y)
        accuracy = svm.score(test_x, test_y)
        train_score = svm.score(train_x, train_y)
        print('=================svm - rbf kernel==================')
        print('test:', accuracy)
        print('train:', train_score)


        log_reg = linear_model.LogisticRegression()
        train_fit = log_reg.fit(train_x, train_y)
        accuracy = log_reg.score(test_x, test_y)
        train_score = log_reg.score(train_x, train_y)
        print('===============logistic regression=================')
        print('test:', accuracy)
        print('train:', train_score)
        print(log_reg.coef_)


        knn = neighbors.KNeighborsClassifier()
        train_fit = knn.fit(train_x, train_y)
        accuracy = knn.score(test_x, test_y)
        train_score = knn.score(train_x, train_y)
        print('=======================knn========================')
        print('test:', accuracy)
        print('train:', train_score)



    predictions = rf.predict(test_x)
    return predictions

def kmean_analysis(inputs, labels):
    #create elbow plot to choose optimal k
    '''
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1,20)

    for k in K:
        #Building and fitting the model
        kmeanModel = cluster.KMeans(n_clusters=k).fit(inputs)
        kmeanModel.fit(inputs)

        distortions.append(sum(np.min(cdist(inputs, kmeanModel.cluster_centers_,
                          'euclidean'),axis=1)) / inputs.shape[0])
        inertias.append(kmeanModel.inertia_)

        mapping1[k] = sum(np.min(cdist(inputs, kmeanModel.cluster_centers_,
                     'euclidean'),axis=1)) / inputs.shape[0]
        mapping2[k] = kmeanModel.inertia_

    print(mapping1.values())
    plt.plot(K, list(mapping1.values()), label = 'distortion')
    #plt.plot(K, list(mapping2.values()), label = 'inertia')
    plt.xlabel = 'number of clusters'
    plt.ylabel = 'distortion/inertia'
    plt.legend()
    plt.title('Elbow Curve')
    plt.show()
    '''
    print('=====================kmeans========================')
    k = 18
    kmeans = cluster.KMeans(n_clusters = k)
    predictions = kmeans.fit_predict(inputs)

    gini_indices = []

    for i in range(k):
        samples = np.argwhere(predictions == i)
        cluster_i = labels[samples]
        cluster_i = cluster_i.reshape((1, -1))[0]
        counts = Counter(list(cluster_i))

        total_num = len(cluster_i)

        max_num = 0
        max_class = 0

        sum_proportion = 0

        for k, v in counts.items():
            if v > max_num:
                max_num = v
                max_class = k
            sum_proportion += (v / total_num) ** 2

        gini_indices.append(1 - sum_proportion)

    fig, ax = plt.subplots()
    im = ax.imshow(np.array(gini_indices).reshape(1, -1))
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('gini index', rotation=-90, va="bottom")
    plt.title('gini index for each cluster')
    plt.show()



def main():
    train_x_file = 'data/primary_training_inputs.csv'
    train_y_file = 'data/training_results.csv'

    test_x_file = 'data/primary_testing_inputs.csv'
    test_y_file = 'data/testing_results.csv'

    train_x, train_y, test_x, test_y = preprocess(train_x_file, train_y_file, test_x_file, test_y_file)
    #train_x, train_y, test_x, test_y = preprocess('data/2014_data.csv', 'data/gov_winners.csv', 'data/2016_data.csv', 'data/pres_winners.csv')
    #iowa_train_x, iowa_train_y, iowa_test_x, iowa_test_y = iowa_data(gov = True)


    predictions = train(train_x, train_y, test_x, test_y)
    #train(iowa_train_x, iowa_train_y, iowa_test_x, iowa_test_y)

    #get predicted results and append to maps data for map visualization
    '''
    test_df = pd.read_csv('data/maps.csv')
    test_df = test_df.assign(predicted=pd.Series(predictions).values)

    test_df.to_csv('data/maps.csv')
    '''

    #combined_iowa = np.concatenate((iowa_train_x, iowa_test_x), axis = 0)
    #combined_iowa_y = np.concatenate((iowa_train_y, iowa_test_y), axis = None)
    #kmean_analysis(combined_iowa, combined_iowa_y)

    #kmean_analysis(train_x, train_y)



if __name__ == '__main__':
    main()
