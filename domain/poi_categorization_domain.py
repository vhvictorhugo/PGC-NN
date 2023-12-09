import pandas as pd
import numpy as np
import json

import spektral.layers
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import spektral as sk
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import sklearn.metrics as skm
from tensorflow.keras import utils as np_utils

from loader.file_loader import FileLoader
from loader.poi_categorization_loader import PoiCategorizationLoader
from extractor.file_extractor import FileExtractor
from model.gnn_base_model_for_transfer_learning import GNNUS_BaseModel
from utils.nn_preprocessing import one_hot_decoding_predicted, top_k_rows,split_graph, top_k_rows_category_user_tracking


class PoiCategorizationDomain:


    def __init__(self, dataset_name):
        self.file_loader = FileLoader()
        self.file_extractor = FileExtractor()
        self.poi_categorization_loader = PoiCategorizationLoader()
        self.dataset_name = dataset_name


    def read_matrix(
            self, 
            adjacency_matrix_filename, 
            temporal_matrix_filename, 
            distance_matrix_filename=None, 
            duration_matrix_filename=None,
            adjacency_regions_matrix_filename=None,
            distance_regions_matrix_filename=None,
            adjacency_regions_feature_filename=None,
    ):

        adjacency_df = self.file_extractor.read_csv(adjacency_matrix_filename).drop_duplicates(subset=['user_id'])
        
        temporal_matrix_df = self.file_extractor.read_csv(temporal_matrix_filename).drop_duplicates(subset=['user_id'])

        if (
            distance_matrix_filename is not None and 
            duration_matrix_filename is not None and
            adjacency_regions_matrix_filename is not None and
            distance_regions_matrix_filename is not None and
            adjacency_regions_feature_filename is not None
        ):
            distance_matrix_df = self.file_extractor.read_csv(distance_matrix_filename).drop_duplicates(
                subset=['user_id'])
            duration_matrix_df = self.file_extractor.read_csv(duration_matrix_filename).drop_duplicates(subset=['user_id'])

            if adjacency_df['user_id'].tolist() != temporal_matrix_df['user_id'].tolist():
                print("\nMATRIZES DIFERENTES\n")
                raise

            adjacency_regions_df = self.file_extractor.read_csv(
                adjacency_regions_matrix_filename
            ).drop_duplicates(subset=['userid'])

            distance_regions_df = self.file_extractor.read_csv(
                distance_regions_matrix_filename
            ).drop_duplicates(subset=['userid'])

            adjacency_regions_feature_df = self.file_extractor.read_csv(
                adjacency_regions_feature_filename
            ).drop_duplicates(subset=['userid'])

            return (
                adjacency_df, 
                temporal_matrix_df, 
                distance_matrix_df, 
                duration_matrix_df, 
                adjacency_regions_df,
                distance_regions_df,
                adjacency_regions_feature_df,
            )
        
        else:
            if adjacency_df['user_id'].tolist() != temporal_matrix_df['user_id'].tolist():
                print("\nMATRIZES DIFERENTES\n")
                raise

            return adjacency_df, temporal_matrix_df

    def _poi_gnn_resize_adjacency_and_category_matrices(
            self, 
            user_matrix, 
            user_matrix_week, 
            user_matrix_weekend, 
            user_category, 
            max_size_matrices, 
            dataset_name
    ):

        more_matrices = 1
        k_original = max_size_matrices
        size = user_matrix.shape[0]
        if size < k_original:
            k = user_matrix.shape[0]
        else:
            k = int(np.floor(size/k_original) * k_original)
        # select the k rows that have the highest sum
        if dataset_name == "user_tracking":
            idx = top_k_rows_category_user_tracking(user_matrix, k, user_category)
        else:
            idx = top_k_rows(user_matrix, k)

        not_used_ids = []
        for i in range(len(idx)):

            if i not in idx:
                not_used_ids.append(i)

        if len(not_used_ids) > 0 or size < max_size_matrices:

            add_more = max_size_matrices - len(not_used_ids)

            count = 0
            i = 0
            for i in idx:

                if count < add_more:

                    not_used_ids.append(i)
                    count += 1

                else:

                    break

        idx = np.array(idx.tolist() + not_used_ids)

        user_matrix = user_matrix[idx[:, None], idx]
        user_matrix_week = user_matrix_week[idx[:, None], idx]
        user_matrix_weekend = user_matrix_weekend[idx[:, None], idx]
        user_category = user_category[idx]


        if k > k_original or len(not_used_ids) > 0:
            k_split = int(np.floor(size/k_original))
            if len(not_used_ids) > 0:
                k_split += 1
            user_matrix = split_graph(user_matrix, k_original, k_split)
            user_matrix_week = split_graph(user_matrix_week, k_original, k_split)
            user_matrix_weekend = split_graph(user_matrix_weekend, k_original, k_split)
            user_category = split_graph(user_category, k_original, k_split)
            idx = split_graph(idx, k_original, k_split)
            more_matrices = k_split            

            return (
                np.array(user_matrix), 
                np.array(user_matrix_week), 
                np.array(user_matrix_weekend), 
                np.array(user_category), 
                np.array(idx), 
                more_matrices
            )

        return (
            np.array([user_matrix]), 
            np.array([user_matrix_week]), 
            np.array([user_matrix_weekend]), 
            np.array([user_category]), 
            np.array([idx]), 
            more_matrices
        )

    def _filter_pmi_matrix(
            self, 
            location_time, 
            location_location, 
            locationid_to_int, 
            visited_location_ids
    ):

        idx = np.array([locationid_to_int[visited_location_ids[i]] for i in range(len(visited_location_ids))])

        location_time = location_time[idx]
        location_location = location_location[idx[:,None], idx].toarray()
        location_location = sk.layers.GCNConv.preprocess(location_location)

        return location_time, location_location

    def poi_gnn_adjacency_preprocessing(
            self,
            inputs,
            max_size_matrices,
            dataset_name,
    ):
        matrices_list = []
        temporal_matrices_list = []
        distance_matrices_list = []
        duration_matrices_list = []
        # week
        matrices_week_list = []
        temporal_matrices_week_list = []
        # weekend
        matrices_weekend_list = []
        temporal_matrices_weekend_list = []
        # location time
        location_time_list = []
        location_location_list = []

        # regions
        adjacency_regions_matrix_list = []
        distance_regions_list = []
        adjacency_regions_feature_list = []

        users_categories = []
        maior = -10

        matrix_df = inputs['all_week']['adjacency']
        ids = matrix_df['user_id'].tolist()
        matrix_df = matrix_df['matrices'].tolist()
        category_df = inputs['all_week']['adjacency']['category'].tolist()
        temporal_df = inputs['all_week']['temporal']['matrices'].tolist()
        distance_df = inputs['all_week']['distance']['matrices'].tolist()
        duration_df = inputs['all_week']['duration']['matrices'].tolist()
        visited_location_ids = inputs['all_week']['adjacency']['visited_location_ids'].tolist()
        location_location_df = inputs['all_week']['location_location']
        location_time_df = inputs['all_week']['location_time'].to_numpy()
        locationid_to_int = inputs['all_week']['int_to_locationid']
        locationid_to_int_ids = locationid_to_int['locationid'].tolist()
        locationid_to_int_ints = locationid_to_int['int'].tolist()
        locationid_to_int = {locationid_to_int_ids[i]: locationid_to_int_ints[i] for i in range(len(locationid_to_int_ints))}
        # week
        matrix_week_df = inputs['week']['adjacency']['matrices'].tolist()
        temporal_week_df = inputs['week']['temporal']['matrices'].tolist()
        # weekend
        matrix_weekend_df = inputs['weekend']['adjacency']['matrices'].tolist()
        temporal_weekend_df = inputs['weekend']['temporal']['matrices'].tolist()

        # regions
        adjacency_regions_matrix_df = inputs['all_week']['adjacency_regions_matrix_df']['matrices'].to_list()
        distance_regions_df = inputs['all_week']['distance_regions_df']['matrices'].to_list()
        adjacency_regions_feature_df = inputs['all_week']['adjacency_regions_feature_df']['matrices'].to_list()

        selected_visited_locations = []

        if len(ids) != len(matrix_df):
            print("\nERRO TAMANHO DA MATRIZ\n")
            exit()

        selected_users = []
        remove = 0

        for i in range(len(ids)):

            number_of_matrices = 1
            user_id = ids[i]

            user_matrices = matrix_df[i]
            user_category = category_df[i]
            user_matrices = json.loads(user_matrices)
            user_matrices = np.array(user_matrices)
            user_category = json.loads(user_category)
            user_category = np.array(user_category)
            # week
            user_matrices_week = matrix_week_df[i]
            user_matrices_week = json.loads(user_matrices_week)
            user_matrices_week = np.array(user_matrices_week)
            # weekend
            user_matrices_weekend = matrix_weekend_df[i]
            user_matrices_weekend = json.loads(user_matrices_weekend)
            user_matrices_weekend = np.array(user_matrices_weekend)
            # user visited
            user_visited = visited_location_ids[i]
            user_visited = json.loads(user_visited)
            user_visited = np.array(user_visited)
            size = user_matrices.shape[0]
            if size > maior:
                maior = size

            # matrices get new size, equal for everyone
            (
                user_matrices, 
                user_matrices_week, 
                user_matrices_weekend, 
                user_category, 
                idxs, 
                number_of_matrices
            ) = (
                self._poi_gnn_resize_adjacency_and_category_matrices(
                    user_matrices, 
                    user_matrices_week, 
                    user_matrices_weekend, 
                    user_category, 
                    max_size_matrices, 
                    dataset_name
                )
            )

            """feature"""
            user_temporal_matrices = temporal_df[i]
            user_temporal_matrices = json.loads(user_temporal_matrices)
            user_temporal_matrices = np.array(user_temporal_matrices)
            # week
            user_temporal_matrices_week = temporal_week_df[i]
            user_temporal_matrices_week = json.loads(user_temporal_matrices_week)
            user_temporal_matrices_week = np.array(user_temporal_matrices_week)
            # weekend
            user_temporal_matrices_weekend = temporal_weekend_df[i]
            user_temporal_matrices_weekend = json.loads(user_temporal_matrices_weekend)
            user_temporal_matrices_weekend = np.array(user_temporal_matrices_weekend)
            """distance"""
            user_distance_matrix = distance_df[i]
            user_distance_matrix = json.loads(user_distance_matrix)
            user_distance_matrix = np.array(user_distance_matrix)
            """duration"""
            user_duration_matrix = duration_df[i]
            user_duration_matrix = json.loads(user_duration_matrix)
            user_duration_matrix = np.array(user_duration_matrix)
            """regions"""
            user_regions_adjacency_matrix = adjacency_regions_matrix_df[i]
            user_regions_adjacency_matrix = json.loads(user_regions_adjacency_matrix)
            user_regions_adjacency_matrix = np.array(user_regions_adjacency_matrix)

            user_distance_regions = distance_regions_df[i]
            user_distance_regions = json.loads(user_distance_regions)
            user_distance_regions = np.array(user_distance_regions)

            user_regions_adjacency_feature = adjacency_regions_feature_df[i]
            user_regions_adjacency_feature = json.loads(user_regions_adjacency_feature)
            user_regions_adjacency_feature = np.array(user_regions_adjacency_feature)

            for i in range(number_of_matrices):

                idx = idxs[i]
                matrices_list.append(sk.layers.ARMAConv.preprocess(user_matrices[i]))
                matrices_week_list.append(sk.layers.ARMAConv.preprocess(user_matrices_week[i]))
                matrices_weekend_list.append(sk.layers.ARMAConv.preprocess(user_matrices_weekend[i]))

                user_temporal_matrix = user_temporal_matrices[idx]
                temporal_matrices_list.append(self._min_max_normalize(user_temporal_matrix))
                user_temporal_matrix_week = user_temporal_matrices_week[idx]
                temporal_matrices_week_list.append(self._min_max_normalize(user_temporal_matrix_week))
                user_temporal_matrix_weekend = user_temporal_matrices_weekend[idx]
                temporal_matrices_weekend_list.append(self._min_max_normalize(user_temporal_matrix_weekend))
                distance_matrices_list.append(user_distance_matrix[idx[:, None], idx])
            
                idx_regions = np.clip(idx, 0, user_regions_adjacency_matrix.shape[0] - 1)
                adjacency_regions_matrix_list.append(
                    user_regions_adjacency_matrix[idx_regions[:, None], idx_regions]
                )

                distance_regions_list.append(
                    user_distance_regions[idx_regions[:, None], idx_regions]
                )

                adjacency_regions_feature_list.append(
                    user_regions_adjacency_feature[idx_regions[:, None], idx_regions]
                )

                duration_matrices_list.append(user_duration_matrix[idx[:, None], idx])
                users_categories.append(user_category[i])
                # location time
                user_location_time, user_location_location = (
                    self._filter_pmi_matrix(
                        location_time_df, 
                        location_location_df, 
                        locationid_to_int, 
                        user_visited[idx]
                    )
                )
                user_location_time = self._min_max_normalize(user_location_time)
                location_time_list.append(user_location_time)
                user_location_location = spektral.layers.ARMAConv.preprocess(user_location_location)
                location_location_list.append(user_location_location)

                for j in user_visited[idx]:
                    selected_visited_locations.append(j)
                    selected_users.append(user_id)

        print("\nQuantidade de usuários", len(ids), " Quantidade removidos: ", remove, "\n")
        self.features_num_columns = temporal_matrices_list[-1].shape[1]
        matrices_list = np.array(matrices_list)
        location_time_list = np.array(location_time_list)
        location_location_list = np.array(location_location_list)
        temporal_matrices_list = np.array(temporal_matrices_list)
        users_categories = np.array(users_categories)

        distance_matrices_list = np.array(distance_matrices_list)
        duration_matrices_list = np.array(duration_matrices_list)

        adjacency_regions_matrix_list = np.array(adjacency_regions_matrix_list)
        distance_regions_list = np.array(distance_regions_list)
        adjacency_regions_feature_list = np.array(adjacency_regions_feature_list)

        # week
        matrices_week_list = np.array(matrices_week_list)
        temporal_matrices_week_list = np.array(temporal_matrices_week_list)

        # weekend
        matrices_weekend_list = np.array(matrices_weekend_list)
        temporal_matrices_weekend_list = np.array(temporal_matrices_weekend_list)
        temporal_matrices_week_list = np.array(temporal_matrices_week_list)

        return (
            users_categories, 
            matrices_list, 
            temporal_matrices_list, 
            distance_matrices_list, 
            duration_matrices_list,
            matrices_week_list, 
            temporal_matrices_week_list, 
            matrices_weekend_list, 
            temporal_matrices_weekend_list,
            location_time_list, 
            location_location_list, 
            selected_users, 
            adjacency_regions_matrix_list,
            distance_regions_list,
            adjacency_regions_feature_list
        )

    def k_fold_split_train_test(
            self,
            k,
            inputs,
            n_splits,
            week_type,
            model_name='poi_gnn'
    ):

        adjacency_list = inputs[week_type]['adjacency']
        temporal_list = inputs[week_type]['temporal']
        user_categories = inputs[week_type]['categories']
        if model_name == "poi_gnn" and week_type == 'all_week':
            distance_list = inputs[week_type]['distance']
            duration_list = inputs[week_type]['duration']
            location_time = inputs[week_type]['location_time']
            location_location_list = inputs[week_type]['location_location']
            adjacency_regions_matrix_list = inputs[week_type]['adjacency_regions_matrix']
            distance_regions_list = inputs[week_type]['distance_regions']
            adjacency_regions_feature_list = inputs[week_type]['adjacency_regions_feature']
        else:
            distance_list = []
            duration_list = []
            location_time = []
            location_location_list = []
            adjacency_regions_matrix_list = []
            distance_regions_list = []
            adjacency_regions_feature_list = []
        skip = False
        if n_splits == 1:
            skip = True
            n_splits = 2
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

        folds = []
        classes_weights = []
        for train_indexes, test_indexes in kf.split(adjacency_list):

            fold, class_weight = (
                self._split_train_test(
                    k,
                    model_name,
                    adjacency_list,
                    user_categories,
                    temporal_list,
                    location_time,
                    location_location_list,
                    distance_list,
                    duration_list,
                    adjacency_regions_matrix_list,
                    distance_regions_list,
                    adjacency_regions_feature_list,
                    train_indexes,
                    test_indexes
                )
            )

            folds.append(fold)
            classes_weights.append(class_weight)
            if skip:
                break

        return folds, classes_weights

    def _split_train_test(
            self,
            k,
            model_name,
            adjacency_list,
            user_categories,
            temporal_list,
            location_time_list,
            location_location_list,
            distance_list,
            duration_list,

            adjacency_regions_matrix_list,
            distance_regions_list,
            adjacency_regions_feature_list,
            
            train_indexes,
            test_indexes
    ):

        size = adjacency_list.shape[0]
        # 'average', 'cv', 'median', 'radius', 'label'
        adjacency_list_train = adjacency_list[train_indexes]
        user_categories_train = user_categories[train_indexes]

        temporal_list_train = temporal_list[train_indexes]

        if len(distance_list) > 0:
            distance_list_train = distance_list[train_indexes]
            duration_list_train = duration_list[train_indexes]
            location_time_list_train = location_time_list[train_indexes]
            location_location_list_train = location_location_list[train_indexes]
            adjacency_regions_matrix_list_train = adjacency_regions_matrix_list[train_indexes]
            distance_regions_list_train = distance_regions_list[train_indexes]
            adjacency_regions_feature_list_train = adjacency_regions_feature_list[train_indexes]
        else:
            distance_list_train = []
            duration_list_train = []
            location_time_list_train = []
            location_location_list_train = []
            adjacency_regions_matrix_list_train = []
            distance_regions_list_train = []
            adjacency_regions_feature_list_train = []

        adjacency_list_test = adjacency_list[test_indexes]
        user_categories_test = user_categories[test_indexes]
        temporal_list_test = temporal_list[test_indexes]

        if len(distance_list) > 0:
            distance_list_test = distance_list[test_indexes]
            duration_list_test = duration_list[test_indexes]
            location_time_list_test = location_time_list[test_indexes]
            location_location_list_test = location_location_list[test_indexes]
            adjacency_regions_matrix_list_test = adjacency_regions_matrix_list[test_indexes]
            distance_regions_list_test = distance_regions_list[test_indexes]
            adjacency_regions_feature_list_test = adjacency_regions_feature_list[test_indexes]
        else:
            distance_list_test = []
            duration_list_test = []
            location_time_list_test = []
            location_location_list_test = []
            adjacency_regions_matrix_list_test = []
            distance_regions_list_test = []
            adjacency_regions_feature_list_test = []

        flatten_train_category = []
        for categories_list in user_categories_train:
            flatten_train_category += categories_list.tolist()
        flatten_train_category = pd.Series(flatten_train_category, name='category')
        flatten_train_category = flatten_train_category.astype('object')
        train_categories_freq = {e:0 for e in flatten_train_category.unique().tolist()}
        for i in range(flatten_train_category.shape[0]):
            train_categories_freq[flatten_train_category.iloc[i]]+=1
        n = sum(train_categories_freq.values())

        total_support = 0
        for e in train_categories_freq:
            total_support+=train_categories_freq[e]

        total_support_inverse = 0
        for e in train_categories_freq:
            total_support_inverse += total_support - train_categories_freq[e]


        for e in train_categories_freq:
            train_categories_freq[e] = (total_support - train_categories_freq[e])/total_support_inverse


        class_weight = list(train_categories_freq.values())
        user_categories_train = np.array([[e for e in row] for row in user_categories_train])
        user_categories_test = np.array([[e for e in row] for row in user_categories_test])

        if len(distance_list) > 0:
            return (
                (
                    adjacency_list_train, 
                    user_categories_train, 
                    temporal_list_train, 
                    distance_list_train, 
                    duration_list_train,
                    location_time_list_train, 
                    location_location_list_train, 
                    adjacency_regions_matrix_list_train,
                    distance_regions_list_train,
                    adjacency_regions_feature_list_train,
                    adjacency_list_test, 
                    user_categories_test, 
                    temporal_list_test, 
                    distance_list_test,
                    duration_list_test, 
                    location_time_list_test, 
                    location_location_list_test,
                    adjacency_regions_matrix_list_test,
                    distance_regions_list_test,
                    adjacency_regions_feature_list_test
                ),
                class_weight
            )
        else:            
            return (
                (
                    adjacency_list_train, 
                    user_categories_train, 
                    temporal_list_train,
                    adjacency_list_test, 
                    user_categories_test, 
                    temporal_list_test
                ), 
                class_weight
            )

    def k_fold_with_replication_train_and_evaluate_model(
            self,
            inputs_folds,
            n_replications,
            max_size_matrices,
            max_size_sequence,
            base_report,
            epochs,
            class_weight,
            country,
            version,
            output_dir
    ):

        folds_histories = []
        folds_reports = []
        models = []
        accuracies = []
        seed = 0
        for i in range(len(inputs_folds['all_week']['folds'])):

            fold = inputs_folds['all_week']['folds'][i]
            fold_week = inputs_folds['week']['folds'][i]
            fold_weekend = inputs_folds['weekend']['folds'][i]
            class_weight = inputs_folds['all_week']['class_weight'][i]
            class_weight_week = inputs_folds['week']['class_weight'][i]
            class_weight_weekend = inputs_folds['weekend']['class_weight'][i]
            histories = []
            reports = []

            for _ in range(n_replications):

                history, report, model, accuracy = self.train_and_evaluate_model(
                    i,
                    fold,
                    fold_week,
                    fold_weekend,
                    class_weight,
                    class_weight_week,
                    class_weight_weekend,
                    max_size_matrices,
                    max_size_sequence,
                    epochs,
                    seed,
                    country,
                    output_dir,
                    version
                )

                seed+=1

                base_report = self._add_location_report(base_report, report)
                histories.append(history)
                reports.append(report)
                models.append(model)
                accuracies.append(accuracy)
            folds_histories.append(histories)
            folds_reports.append(reports)

        best_model = self._find_best_model(models, accuracies)

        return folds_histories, base_report, best_model

    def train_and_evaluate_model(
            self,
            fold_number,
            fold,
            fold_week,
            fold_weekend,
            class_weight,
            class_weight_week,
            class_weight_weekend,
            max_size_matrices,
            max_size_sequence,
            epochs,
            seed,
            country,
            output_dir,
            version="normal",
            model=None
    ):
        (adjacency_train, 
        y_train, 
        temporal_train, 
        distance_train, 
        duration_train,  
        location_time_train, 
        location_location_train,
        adjacency_regions_matrix_train,
        distance_regions_list_train,
        adjacency_regions_feature_train,
        adjacency_test, 
        y_test, 
        temporal_test, 
        distance_test, 
        duration_test, 
        location_time_test, 
        location_location_test,
        adjacency_regions_matrix_test,
        distance_regions_list_test,
        adjacency_regions_feature_test) = fold

        (adjacency_week_train, 
        y_train_week, 
        temporal_train_week, 
        adjacency_test_week, 
        y_test_week, 
        temporal_test_week) = fold_week

        (adjacency_train_weekend, 
        y_train_weekend, 
        temporal_train_weekend,
        adjacency_test_weekend, 
        y_test_weekend, 
        temporal_test_weekend) = fold_weekend

        max_total = 0
        max_user = -1

        for i in range(len(adjacency_test)):
            user_total = np.sum(adjacency_test[i])
            if user_total > max_total:
                max_total = user_total
                max_user = i

        num_classes = max(y_train.flatten()) + 1
        max_size = max_size_matrices
        lr = 0.001
        print("\nQuantidade de classes: ", num_classes)
        print("\nTamanho maximo", max_size_matrices)

        print("\nTamanho das matrizes de treino: ", 
            adjacency_train.shape, 
            temporal_train.shape,
            adjacency_week_train.shape, 
            temporal_train_week.shape, 
            distance_train.shape, 
            duration_train.shape, 
            location_time_train.shape, 
            location_location_train.shape,
            adjacency_regions_matrix_train.shape,
            distance_regions_list_train.shape,
            adjacency_regions_feature_train.shape
        )

        print("\nTamanho das matrizes de teste: ", 
                adjacency_test.shape, 
                temporal_test.shape,
                adjacency_test_week.shape, 
                temporal_test_week.shape, 
                distance_test.shape, 
                duration_test.shape, 
                location_time_test.shape, 
                location_location_test.shape,
                adjacency_regions_matrix_test.shape,
                distance_regions_list_test.shape,
                adjacency_regions_feature_test.shape
        )
        
        model = (
            GNNUS_BaseModel(
                num_classes,
                max_size,
                max_size_sequence,
                self.features_num_columns
            )
            .build(seed=seed)
        )
        
        batch = max_size * 2

        print("\nTamanho do batch: ", batch)

        user_index = max_user
        self.heatmap_matrices(
            str(fold_number), 
            [
                adjacency_test[user_index], 
                adjacency_test_week[user_index], 
                adjacency_test_weekend[user_index],
                temporal_test[user_index], 
                temporal_test_week[user_index], 
                temporal_test_weekend[user_index], 
                location_time_test[user_index], 
                location_location_test[user_index],
                adjacency_regions_matrix_test[user_index],
                distance_regions_list_train[user_index],
                adjacency_regions_feature_test[user_index],
            ],
            [
                "Adjacency", 
                "Adjacency (weekday)", 
                "Adjacency (weekend)", 
                "Temporal", 
                "Temporal (weekday)", 
                "Temporal (weekend)", 
                "Location_time", 
                "Location_location"
            ],
            output_dir
        )

        input_train = [
            adjacency_train, 
            adjacency_week_train, 
            adjacency_train_weekend,
            temporal_train, 
            temporal_train_week, 
            temporal_train_weekend, 
            distance_train,
            duration_train, 
            location_time_train, 
            location_location_train,
            adjacency_regions_matrix_train,
            distance_regions_list_train,
            adjacency_regions_feature_train
        ]

        input_test = [
            adjacency_test, 
            adjacency_test_week, 
            adjacency_test_weekend, 
            temporal_test, 
            temporal_test_week, 
            temporal_test_weekend,
            distance_test, 
            duration_test, 
            location_time_test, 
            location_location_test,
            adjacency_regions_matrix_test,
            distance_regions_list_test,
            adjacency_regions_feature_test
        ]
        
        # verifying whether categories arrays are equal
        compare1 = y_train == y_train_week
        compare2 = y_train_week == y_train_weekend
        compare3 = y_test == y_test_week
        compare4 = y_test_week == y_test_weekend
        if not(compare1.all() and compare2.all() and compare3.all() and compare4.all()):
            print("\nListas difernetes de categorias\n")
            exit()

        model.compile(
            optimizer=Adam(learning_rate=lr), 
            loss=['categorical_crossentropy'],
            weighted_metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")]
        )

        y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes=num_classes)

        hi = model.fit(
            x=input_train,
            y=y_train, 
            validation_data=(input_test, y_test),
            epochs=epochs, 
            batch_size=batch,
            shuffle=False,  # Shuffling data means shuffling the whole graph
            callbacks=[EarlyStopping(patience=100, restore_best_weights=True)]
        )        

        h = hi.history
        y_predict_location = model.predict(input_test, batch_size=batch)

        scores = model.evaluate(input_test, y_test, batch_size=batch)
        print("\nscores: ", scores)

        # To transform one_hot_encoding to list of integers, representing the locations
        y_predict_location = one_hot_decoding_predicted(y_predict_location)
        y_test = one_hot_decoding_predicted(y_test)
        report = skm.classification_report(y_test, y_predict_location, output_dict=True)
        print(report)

        return h, report, model, report['accuracy']

    def heatmap_matrices(self, fold_number, matrices, names, output_dir):

        for matrix, name in zip(matrices, names):

            self.poi_categorization_loader.heatmap(output_dir, matrix, name.replace(" ", "_")+"_"+fold_number, name, (10,10), True)

    def _add_location_report(self, location_report, report):
        for l_key in report:
            if l_key == 'accuracy':
                location_report[l_key].append(report[l_key])
                continue
            for v_key in report[l_key]:
                location_report[l_key][v_key].append(report[l_key][v_key])

        return location_report

    def _find_best_model(self, models, accuracies):

        index = np.argmax(accuracies)
        return models[index]

    def preprocess_report(self, report, int_to_categories):

        new_report = {}

        for key in report:
            if key != 'accuracy' and key != 'macro avg' and key != 'weighted avg':
                new_report[int_to_categories[key]] = report[key]
            else:
                new_report[key] = report[key]

        return new_report

    def _min_max_normalize(self, matrix):

        matrix_1 = matrix.transpose()
        scaler = MinMaxScaler()
        scaler.fit(matrix_1)
        matrix_1 = scaler.transform(matrix_1).transpose()

        return matrix_1        