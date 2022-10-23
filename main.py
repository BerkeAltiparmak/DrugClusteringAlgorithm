import pandas as pd
from pandas import DataFrame
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import timeit
import math

import weighted_graph_creator as wgc
import graph_visualization as gv
from sklearn.mixture import GaussianMixture



def get_df() -> DataFrame:
    """
    Get the pandas dataframe by reading from a csv file.
    Select only the 'auc', 'name', 'ccle_name' columns.
    If an auc value is greater than 1, make it 1.
    :return: df with 'auc', 'name', 'ccle_name' columns.
    """
    df = pd.read_csv('secondary-screen-dose-response-curve-parameters.csv')
    df_wanted = df[['auc', 'name', 'ccle_name', 'moa']]
    df_responsive = df_wanted
    df_responsive.loc[df_responsive['auc'] > 1, 'auc'] = 1

    return df_responsive


def plot_heatmap(heatmap, xlim=0, ylim=0):
    """
    Draw the heatmap using matplotlib and seaborn
    :param heatmap: the heatmap with 'auc', 'name', 'ccle_name' columns.
    :param xlim: x axis limit
    :param ylim: y axis limit
    :return: the drawing of the heatmap
    """
    fig, ax = plt.subplots()
    sns.set()
    sns.heatmap(heatmap, cmap='RdBu')

    if xlim + ylim > 0:
        plt.xlim(right=xlim)
        plt.ylim(bottom=ylim)
        plt.xticks(np.arange(0.5, xlim, step=1), [x for x in range(0, xlim)], rotation=0)
        plt.yticks(np.arange(0.5, ylim, step=1), [x for x in range(0, ylim)], rotation=0)
    # plt.xticks(rotation=15)
    # plt.yticks(rotation=15)
    plt.show()


def plot_histogram(lst):
    series = pd.Series(lst)
    series.plot.hist(grid=True, bins=40, rwidth=0.95, color='#607c8e')
    plt.title('Histogram')
    plt.xlabel('R Values')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.75)


def plot_comparison(x, y):
    plt.plot(x, y)


def get_useful_data(hm, auc_cutoff=0.5):
    """
    Remove drugs that do not have a single response from any of the cell lines.
    That is, remove any drug that doesn't have a single auc value of less than 0.5.
    :param hm: the heatmap
    :return: the heatmap without the drugs that do not have a single auc value of less than 0.5.
    """
    hm.reset_index()
    useful_drugs_list = []

    for index, row in hm.iterrows():
        if any(auc <= auc_cutoff for auc in row):
            useful_drugs_list.append(index)
    new_hm = hm[hm.index.isin(useful_drugs_list)]

    return (new_hm, useful_drugs_list)


def get_similarity_labeling_matrix(corr_m, drug_list, r_cutoff=0.7):
    initial_cluster = []
    for row_i in corr_m:
        row_i_similardrugs_set = []
        for drug_j_index in range(0, len(row_i)):
            drug_j_similarity_with_drug_i = row_i[drug_j_index]
            if drug_j_similarity_with_drug_i >= r_cutoff:
                row_i_similardrugs_set.append(drug_list[drug_j_index])
        initial_cluster.append(row_i_similardrugs_set)

    return initial_cluster


def get_gaussian_mixture_model(drug_lit, k):
    gmm = GaussianMixture(n_components=k)
    gmm.fit(drug_lit)
    """
    yP = gmm.predict_proba(XB)  # produces probabilities
    # Arbitrary labels with unsupervised clustering may need to be reversed
    if len(XB[np.round(yP[:, 0]) != yB]) > n / 4: yP = 1 - yP
    """
    return gmm


def get_similar_drugs(initial_cluster, has_outcast_cluster=False):
    clustered_drugs = []
    outcast_cluster = []
    for label in initial_cluster:
        if len(label) > 1 and label not in clustered_drugs:
            clustered_drugs.append(label)
        elif len(label) == 1 and has_outcast_cluster:
            outcast_cluster.extend(label)

    if has_outcast_cluster and len(outcast_cluster) > 0:
        clustered_drugs.append(outcast_cluster)

    return clustered_drugs


def compare_r_cutoffs(corr_m, drug_list, r_cutoff_list, has_outcast_cluster_for_all=False):
    num_sets_list = []
    total_drugs_list = []
    rcutoff_clustered_drugs_list = []
    for r_cutoff in r_cutoff_list:
        initial_cluster = get_similarity_labeling_matrix(corr_m, drug_list, r_cutoff)
        clustered_drugs = get_similar_drugs(initial_cluster, has_outcast_cluster=has_outcast_cluster_for_all)
        num_sets = len(clustered_drugs)
        total_drugs = len(set([item for sublist in clustered_drugs for item in sublist]))
        num_sets_list.append(num_sets)
        total_drugs_list.append(total_drugs)
        rcutoff_clustered_drugs_list.append(clustered_drugs)

    return (num_sets_list, total_drugs_list, rcutoff_clustered_drugs_list)


def get_drug_label_map(df, drug_list):
    drug_label_map = {}
    for drug_name in drug_list:
        label = df.loc[df['name'] == drug_name, 'moa'].iloc[0]
        drug_label_map[drug_name] = label

    return drug_label_map


def get_label_count_map(drug_list, drug_label_map):
    label_count_map = {}
    for drug_name in drug_list:
        label = drug_label_map[drug_name]
        if label in label_count_map:
            label_count_map[label] += 1
        else:
            label_count_map[label] = 1

    return label_count_map


def get_class_labels_entropy(total_nonunique_labels, drugs_label_count_map):
    # total_nonunique_labels = len(drug_classes_map)
    H_y = 0
    for label in drugs_label_count_map:
        label_prop = drugs_label_count_map[label] / total_nonunique_labels
        H_y += -label_prop * math.log2(label_prop)
    return H_y


def get_cluster_labels_entropy(total_nonunique_labels, clustered_drugs):
    H_c = 0
    for cluster in clustered_drugs:
        cluster_prop = len(cluster) / total_nonunique_labels
        if(cluster_prop <= 0):
            print("olm")
            print(cluster_prop)
            print(cluster)
            print(len(cluster))
        H_c += -cluster_prop * math.log2(cluster_prop)
    return H_c


def get_mutual_information_between_classes_and_clusters(class_labels_entropy, drug_classes_map, total_nonunique_labels, clustered_drugs):
    H_y_c = 0
    H_y = class_labels_entropy
    for cluster in clustered_drugs:
        cluster_prop = len(cluster) / total_nonunique_labels
        label_count_within_cluster_map = {}
        for drug in cluster:
            label = drug_classes_map[drug]
            if label in label_count_within_cluster_map:
                label_count_within_cluster_map[label] += 1
            else:
                label_count_within_cluster_map[label] = 1
        H_y_within_cluster = get_class_labels_entropy(len(cluster), label_count_within_cluster_map)
        H_y_c += cluster_prop * H_y_within_cluster

    I_y_c = H_y - H_y_c

    return I_y_c


def calculate_normalized_mutual_information(drug_label_map, clustered_drugs):
    """
    NMI = 2 * / (class_labels_entropy + cluster_labels_entropy)
    :param df:
    :param clustered_drugs:
    :return:
    """
    all_drugs_in_cluster = [item for sublist in clustered_drugs for item in sublist]
    label_count_map = get_label_count_map(all_drugs_in_cluster, drug_label_map)
    total_nonunique_labels = len(all_drugs_in_cluster)
    H_y = get_class_labels_entropy(total_nonunique_labels, label_count_map)
    H_c = get_cluster_labels_entropy(total_nonunique_labels, clustered_drugs)
    I_y_c = get_mutual_information_between_classes_and_clusters(H_y, drug_label_map, total_nonunique_labels, clustered_drugs)

    NMI_y_c = 0
    if H_y * H_c > 0:
        NMI_y_c = 2 * I_y_c / (H_y + H_c)

    return NMI_y_c


def compare_clusters_normalized_mutual_information(drug_label_map, rcutoff_clustered_drugs_list):
    nmi_list = []
    for clustered_drugs in rcutoff_clustered_drugs_list:
        nmi = calculate_normalized_mutual_information(drug_label_map, clustered_drugs)
        nmi_list.append(nmi)

    return nmi_list


if __name__ == '__main__':
    start = timeit.default_timer()
    df = get_df()
    print('dataframe in ', timeit.default_timer() - start)

    heatmap = pd.pivot_table(df, values='auc', index=['name'], columns='ccle_name')
    hm, useful_drug_list = get_useful_data(heatmap, 1)  # 1 for including everything
    hm.fillna(0.5, inplace=True)
    print('heatmap ready in ', timeit.default_timer() - start)
    # print(hm)
    # kmeans = get_clusters(heatmap, 5)
    # draw_heatmap(hm)

    corr_m = np.corrcoef(hm)
    # all_r_values_for_histogram = [item for sublist in corr_m for item in sublist]
    # plot_histogram(flat_list)
    print('correlation matrix ready in ', timeit.default_timer() - start)

    initial_cluster = get_similarity_labeling_matrix(corr_m, useful_drug_list, 0.6)
    clustered_drugs = get_similar_drugs(initial_cluster, has_outcast_cluster=True)
    print('single correlation data ready in ', timeit.default_timer() - start)

    r_cutoff_list = np.arange(0, 1, 0.05)
    num_sets, total_drugs, rcutoff_clustered_drugs_list = compare_r_cutoffs(
        corr_m, useful_drug_list, r_cutoff_list, has_outcast_cluster_for_all=True)
    print('multiple correlation data ready in ', timeit.default_timer() - start)
    # plot_comparison(r_cutoff_list, np.divide(total_drugs, num_sets))

    #drug_label_map = get_drug_label_map(df, useful_drug_list)
    #nmi_list = compare_clusters_normalized_mutual_information(drug_label_map, rcutoff_clustered_drugs_list)
    #plot_comparison(r_cutoff_list, nmi_list)
    #print('nmi comparison ready in ', timeit.default_timer() - start)

    """
    cluster_graph = wgc.get_weighted_graph_from_clusters(clustered_drugs, has_outcast_cluster=True)
    gv.visualize_graph(cluster_graph)
    """

    """
    multiple_drug_occurance = {}
    cluster_length_list = []
    for c in clustered_drugs:
        sum_of_drugs_in_c = 0
        for d in c:
            if d not in multiple_drug_occurance:
                multiple_drug_occurance[d] = 1
            else:
                multiple_drug_occurance[d] += 1
            sum_of_drugs_in_c += 1
        cluster_length_list.append(sum_of_drugs_in_c)
    """


