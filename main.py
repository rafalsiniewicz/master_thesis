from pso import *
from clustering import *
from base_class import *
if __name__ == "__main__":
    c = Corpus()
    c.read_documents()
    c.get_vocabulary()
    c.tf_idf()
    features = []
    # for k in range(len(c.documents)):
    #     features.append(c.documents[k].features)
    for k in range(len(c.documents)):
        pso = PSO(features=c.documents[k].features, max_iter=1)
        pso.set_particles_parameters()
        pso.initialize_swarm()
        # for i in pso.swarm:
        #     print(i)

        pso.run()

    # for i in PSO.selected_features:
    #     # print(i)
    #     print(len(i))
    kmeans = Clustering(features=PSO.selected_features)
    kmeans.run()
    kmeans.validate_clustering()
    # for i in PSO.selected_features:
    #     print(i)
