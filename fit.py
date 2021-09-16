from som import SOM
import numpy as np
from dev.Grad_norm import Grad_Norm
import pickle


if __name__ == '__main__':

    keyword = "ファッション"
    model = "SOM"
    feature_file = 'data/tmp/'+keyword+'.npy'
    label_file = 'data/tmp/'+keyword+'_label.npy'
    X = np.load(feature_file)
    labels_long = np.load(label_file)
    labels = ["{:.8}".format(label.replace(keyword,'')) for label in labels_long]
    print(labels)

    nb_epoch = 50
    resolution = 10
    sigma_max = 2.2
    sigma_min = 0.3
    tau = 50
    latent_dim = 2
    seed = 1

    title_text= "animal map"
    umat_resolution = 100 # U-matrix表示の解像度

    np.random.seed(seed)

    som = SOM(X, latent_dim=latent_dim, resolution=resolution, sigma_max=sigma_max, sigma_min=sigma_min, tau=tau, init='PCA')
    som.fit(nb_epoch=nb_epoch)

    som_umatrix = Grad_Norm(X=X,
                            Z=som.history['z'],
                            sigma=som.history['sigma'],
                            labels=labels,
                            resolution=umat_resolution,
                            title_text=title_text)
    som_umatrix.draw_umatrix()


    # with open('data/tmp/'+keyword+'_'+model+'.pickle', 'wb') as f:
    #     pickle.dump(som, f)
