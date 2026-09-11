from preprocessing import Preprocessing as pre


X_train, Y_train,X_val, Y_val, X_test, Y_test,adj = pre(data_path='Dataset/metr-la.csv',adj_path='Dataset/adj_Metr-LA.pkl').normalization()