# %%
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


# %%
def relu(x):
    return x * (x > 0)

def prepare_data(encoder, layers, data):
    for l in range(layers):
        data = relu(data @ encoder.coefs_[l] + encoder.intercepts_[l])

    return data

# %%
RANDOM_SEED = 1410
N_SAMPLES  = 1000

X, y = make_classification(random_state=RANDOM_SEED, class_sep=.75, n_samples=N_SAMPLES)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

# %%
nn_clf = MLPClassifier(hidden_layer_sizes=(1000, 500, 100, 50, 20, 10, 5), random_state=RANDOM_SEED)
tr_clf = DecisionTreeClassifier(random_state=RANDOM_SEED)

# %%
nn_clf.fit(X_train, y_train)
nn_clf.score(X_test, y_test)
# %%
scores = {}
clf = clone(tr_clf)
clf.fit(X_train, y_train)
res = clf.score(X_test, y_test)
scores['No encoding'] = res

for layers_num in range(7):
    X_train_encoded = prepare_data(nn_clf, layers_num + 1, X_train)
    X_test_encoded = prepare_data(nn_clf, layers_num + 1, X_test)
    
    clf = clone(tr_clf)
    clf.fit(X_train_encoded, y_train)
    res = clf.score(X_test_encoded, y_test)
    scores[f'Layer {layers_num + 1}'] = res

# %%
scores

# %%
