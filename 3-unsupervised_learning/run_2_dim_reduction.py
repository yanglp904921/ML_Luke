
import pickle as pkl
import utils_load as load
import utils_cal as ucal


data_list = ['mnist']
# data_list = ['phish', 'mnist']
for data in data_list:
    if data == 'phish':
        x, y, _ = load.load_phish_data()
    elif data == 'mnist':
        x, y, _ = pkl.load(file=open('data/{}.pkl'.format(data), 'rb'))
        x, y = x[0:20000], y[0:20000]

    df_pca = ucal.search_pca(x)
    df_ica = ucal.search_ica(x)
    df_rca = ucal.search_rca(x)
    df_rfc = ucal.search_rfc(x, y)
    x_pca = ucal.transform_x_pca(x, df_pca)
    x_ica = ucal.transform_x_ica(x, df_ica)
    x_rca = ucal.transform_x_rca(x)
    x_rfc = ucal.transform_x_rfc(x, df_rfc)
    pkl.dump([df_pca, df_ica, df_rca, df_rfc, x_pca, x_ica, x_rca, x_rfc],
             file=open('results/dim_reduction_{}.pkl'.format(data), 'wb'))

