from pso import *
from clustering import *
from base_class import *
import time
import xlsxwriter
import os

pso_parameters = {"max_iter": [10, 20, 50, 100], "num_features_select": [1, 2, 3, 5, 10, 20],
                  "num_particles": [10, 20, 50, 100], "w": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                  "c1": [1.5, 1.6, 1.7, 1.8, 1.9, 2], "c2": [1.5, 1.6, 1.7, 1.8, 1.9, 2]
                  }
k_means_parameters = {"k_means_max_iter": [5, 10, 20, 50], "k_means_n_init": [10, 20, 50, 100, 200]}

number_of_iterations = 10
selected_parameters = {"max_iter": pso_parameters["max_iter"][0],
                       "num_features_select": pso_parameters["num_features_select"][0],
                       "num_particles": pso_parameters["num_particles"][0], "w": pso_parameters["w"][0],
                       "c1": pso_parameters["c1"][0], "c2": pso_parameters["c2"][0],
                       "k_means_max_iter": k_means_parameters["k_means_max_iter"][0],
                       "k_means_n_init": k_means_parameters["k_means_n_init"][0]}

selected_key = "max_iter"
select_features = True
instance_folder = "instance1"
test = True
if select_features:
    if test:
        folder_name = 'results\\test\\with_selection\\{}'.format(selected_key)
    else:
        folder_name = 'results/{}/with_selection/{}'.format(instance_folder, selected_key)
else:
    if test:
        folder_name = 'results/test/without_selection/{}'.format(selected_key)
    else:
        folder_name = 'results/{}/without_selection/{}'.format(instance_folder, selected_key)

full_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), folder_name)
if not os.path.exists(full_path):
    os.mkdir(full_path)

if __name__ == "__main__":
    workbook = xlsxwriter.Workbook(folder_name + '/{}.xlsx'.format(selected_key))
    for e, parameter in enumerate(pso_parameters[selected_key]):
        average_results = {"time_elapsed": 0, "silhouette_score": 0, "v_measure_score": 0, "adjusted_rand_score": 0,
                           "davies_bouldin_score": 0}
        worksheet = workbook.add_worksheet('{}={}'.format(selected_key, parameter))
        row = 0
        column = 0
        for k, v in selected_parameters.items():
            worksheet.write(row, column, k)
            worksheet.write(row, column + 1, v)
            row += 1
        row += 1

        for iteration in range(number_of_iterations):
            start = time.time()
            PSO.selected_features = []
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
            if select_features:
                for k in range(len(c.documents)):
                    pso = PSO(features=c.documents[k].features, max_iter=parameter,
                              num_features_select=selected_parameters["num_features_select"],
                              num_particles=selected_parameters["num_particles"], evaluate_function='MAD')
                    pso.set_particles_parameters(w=selected_parameters["w"], c1=selected_parameters["c1"],
                                                 c2=selected_parameters["c2"])
                    # pso.set_particles_parameters(w=10, c1=1, c2=20)
                    pso.initialize_swarm()
                    # for i in pso.swarm:
                    #     print(i)

                    pso.run(show_logs=False)
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
                kmeans_features = PSO.selected_features
            else:
                kmeans_features = features

            # for i in PSO.selected_features:
            #     # print(i)
            #     print(len(i))
            kmeans = Clustering(features=kmeans_features, n_clusters=3,
                                max_iter=selected_parameters["k_means_max_iter"],
                                n_init=selected_parameters["k_means_n_init"], init_type='k-means++')
            # kmeans = Clustering(features=features, n_clusters=3, max_iter=100, n_init=100, init_type='k-means++')
            # kmeans.determine_optimal_number_of_clusters(method='silhoulette_method')
            # kmeans.elbow_method()
            # kmeans.silhoulette_method()
            # kmeans.davies_bouldin_method()

            kmeans.run(sklearn_method=False, show_logs=True)
            validation_scores = kmeans.validate_clustering()
            # kmeans.run()
            # kmeans.validate_clustering()

            f = open(
                folder_name + "/results_{}={}_iter={}.txt".format(selected_key, parameter, iteration),
                "w")
            f.write("######## SELECTED PARAMETERS VALUES ########\n")
            for k, v in selected_parameters.items():
                f.write("{k} = {v}\n".format(k=k, v=v))
            f.write("############################################\n")
            print("Document:{}\tLabel:".format(" " * (len(c.get_longest_filename()) - len("Document"))))
            f.write("Document:{}\tLabel:\n".format(" " * (len(c.get_longest_filename()) - len("Document"))))
            for i, d in enumerate(c.documents):
                print("{}{}\t{}".format(d.filename, " " * (len(c.get_longest_filename()) - len(d.filename)),
                                        kmeans.labels[i]))
                f.write("{}{}\t{}\n".format(d.filename, " " * (len(c.get_longest_filename()) - len(d.filename)),
                                            kmeans.labels[i]))

            # for i in PS
            #
            # O.selected_features:
            #     print(i)
            end = time.time()
            if iteration == 0:
                worksheet.write(row, column, "Iteration")
            worksheet.write(row + 1 + iteration, column, "{}".format(iteration))
            column += 1
            print("Time elapsed: ", end - start)  # CPU seconds elapsed (floating point)
            f.write("Time elapsed: {}\n".format(end - start))
            if iteration == 0:
                worksheet.write(row, column, "Time elapsed")
            worksheet.write(row + 1 + iteration, column, "{}".format(end - start).replace(".", ","))
            column += 1
            f.write("Validation scores:\n")
            for k, v in validation_scores.items():
                f.write("{k} = {v}\n".format(k=k, v=v))
                average_results[k] += v
                if iteration == 0:
                    worksheet.write(row + iteration, column, k)
                worksheet.write(row + 1 + iteration, column, v)
                column += 1
            f.close()
            average_results["time_elapsed"] += (end - start)
            column = 0

        avg_f = open(folder_name + "/results_{}={}_average.txt".format(selected_key, parameter), "w")
        avg_f.write("######## SELECTED PARAMETERS VALUES ########\n")
        for k, v in selected_parameters.items():
            avg_f.write("{k} = {v}\n".format(k=k, v=v))
        avg_f.write("############################################\n")
        avg_sum = 0
        worksheet.write(row + 1 + number_of_iterations, column, "Average")
        column += 1
        for k, v in average_results.items():
            avg_v = v / number_of_iterations
            avg_f.write("{k} = {avg_v}\n".format(k=k, avg_v=avg_v))
            worksheet.write(row + 1 + number_of_iterations, column, avg_v)
            if k not in ["davies_bouldin_score", "time_elapsed"]:
                avg_sum += avg_v
            column += 1
        row += 1
        column = 0
        worksheet.write(row + 1 + number_of_iterations, column,
                        "Sum of average silhouette_score, v_measure_score, adjusted_rand_score")
        worksheet.write(row + 1 + number_of_iterations, column + 1, avg_sum)
        avg_f.write("Sum of average silhouette_score, v_measure_score, adjusted_rand_score = {}".format(avg_sum))
        avg_f.close()

    workbook.close()
