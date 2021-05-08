from pso import *
from clustering import *
from base_class import *
if __name__ == "__main__":
    c = Corpus()
    c.read_documents()
    c.get_vocabulary()
    c.tf_idf()
    features = []
    for k in range(len(c.documents)):
        features.append(c.documents[k].features)

    # fs = {}
    # for i in range(len(c.documents)):
    #     fs[c.documents[i].filename] = features[i]
    # print("ficzers")
    # for i in fs:
    #     print(i, fs[i])
    # input()


    for k in range(len(c.documents)):
        pso = PSO(features=c.documents[k].features, max_iter=1, num_features_select=3)
        pso.set_particles_parameters()
        pso.initialize_swarm()
        # for i in pso.swarm:
        #     print(i)

        pso.run()
        c.documents[k].selected_features = PSO.selected_features[k]

    # sel_fs = {}
    # for i in range(len(c.documents)):
    #     sel_fs[c.documents[i].filename] = pso.selected_features[i]
    #
    #
    # print("selected ficzers")
    # for i in sel_fs:
    #     print(i, sel_fs[i])
    # input()

    # print("selected ficzers")
    # for i in range(len(c.documents)):
    #     print(c.documents[i].filename, c.documents[i].features)
    # input()
    fs_selected = [c.documents[i].selected_features for i in range(len(c.documents))]

    # for i in PSO.selected_features:
    #     # print(i)
    #     print(len(i))
    kmeans = Clustering(features=PSO.selected_features, n_clusters=3, max_iter=100, n_init=10, init_type='k-means++')
    # kmeans = Clustering(features=features, n_clusters=3)
    kmeans.run(sklearn_method=False)
    kmeans.run()
    kmeans.validate_clustering()
    print("Document\tlabel")
    for i, d in enumerate(c.documents):
        print("{}\t{}".format(d.filename, kmeans.labels[i]))
    # for i in PSO.selected_features:
    #     print(i)
