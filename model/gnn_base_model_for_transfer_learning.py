import tensorflow as tf
from spektral.layers.convolutional import ARMAConv
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model


iterations = 1          # Number of iterations to approximate each ARMA(1)
order = 1               # Order of the ARMA filter (number of parallel stacks)
share_weights = True    # Share weights in each ARMA stack
dropout = 0.5           # Dropout rate applied between layers
dropout_skip = 0.3    # Dropout rate for the internal skip connection of ARMA
l2_reg = 5e-5           # L2 regularization rate
learning_rate = 1e-2    # Learning rate
epochs = 15          # Number of training epochs
es_patience = 100       # Patience for early stopping

class GNNUS_BaseModel:

    def __init__(self, classes, max_size_matrices, max_size_sequence, features_num_columns: int):
        self.max_size_matrices = max_size_matrices
        self.max_size_sequence = max_size_sequence
        self.classes = classes
        self.features_num_columns = features_num_columns

    def build(self, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)

        A_input = Input((self.max_size_matrices,self.max_size_matrices))
        A_week_input = Input((self.max_size_matrices, self.max_size_matrices))
        A_weekend_input =  Input((self.max_size_matrices, self.max_size_matrices))
        Temporal_input = Input((self.max_size_matrices, self.features_num_columns))
        Temporal_week_input = Input((self.max_size_matrices, 24))
        Temporal_weekend_input = Input((self.max_size_matrices, 24))
        Distance_input = Input((self.max_size_matrices,self.max_size_matrices))
        Duration_input = Input((self.max_size_matrices,self.max_size_matrices))
        A_week_input = Input((self.max_size_matrices, self.max_size_matrices))
        Location_time_input = Input((self.max_size_matrices, self.features_num_columns))
        Location_location_input = Input((self.max_size_matrices, self.max_size_matrices))

        out_temporal = ARMAConv(
            20, activation='elu',
            gcn_activation='gelu', 
            share_weights=share_weights,
            dropout_rate=dropout_skip
        )([Temporal_input, A_input])
        out_temporal = Dropout(0.3)(out_temporal)
        out_temporal = ARMAConv(self.classes, activation="softmax")([out_temporal, A_input])

        out_week_temporal = ARMAConv(
            20, 
            activation='elu',
            gcn_activation='gelu', 
            share_weights=share_weights,
            dropout_rate=dropout_skip
        )([Temporal_week_input, A_week_input])
        out_week_temporal = Dropout(0.3)(out_week_temporal)
        out_week_temporal = ARMAConv(
            self.classes, 
            activation="softmax"
        )([out_week_temporal, A_week_input])

        out_weekend_temporal = ARMAConv(
            20, 
            activation='elu',
            gcn_activation='gelu', 
            share_weights=share_weights,
            dropout_rate=dropout_skip
        )([Temporal_weekend_input, A_weekend_input])
        out_weekend_temporal = Dropout(0.3)(out_weekend_temporal)
        out_weekend_temporal = ARMAConv(
            self.classes,
            activation="softmax"
        )([out_weekend_temporal, A_weekend_input])

        out_distance = ARMAConv(
            20, 
            activation='elu',
            gcn_activation='gelu'
        )([Distance_input, A_input])
        out_distance = Dropout(0.3)(out_distance)
        out_distance = ARMAConv(self.classes, activation="softmax")([out_distance, A_input])

        out_duration = ARMAConv(
            20, 
            activation='elu',
            gcn_activation='gelu'
        )([Duration_input, A_input])
        out_duration = Dropout(0.3)(out_duration)
        out_duration = ARMAConv(
            self.classes,
            activation="softmax"
        )([out_duration, A_input])

        # usa
        out_location_location = ARMAConv(
            20, 
            activation='elu',
            gcn_activation='gelu'
        )([Location_time_input, Location_location_input])
        out_location_location = Dropout(0.3)(out_location_location)
        out_location_location = ARMAConv(
            self.classes,
            activation="softmax"
        )([out_location_location, Location_location_input])

        # usa
        out_location_time = Dense(40, activation='relu')(Location_time_input)
        out_location_time = Dense(self.classes, activation='softmax')(out_location_time)

        out_dense = (
            tf.Variable(2.) *  out_location_location 
            + tf.Variable(2.) *  out_location_time
        )
        out_dense = Dense(self.classes, activation='softmax')(out_dense)

        out_gnn = (
            tf.Variable(1.) * out_temporal 
            + tf.Variable(1.) * out_week_temporal 
            + tf.Variable(1.) * out_weekend_temporal 
            + tf.Variable(1.) * out_distance 
            + tf.Variable(1.) * out_duration
        )

        out_gnn = Dense(self.classes, activation='softmax')(out_gnn)
        out = (tf.Variable(1.) * out_dense + tf.Variable(1.) * out_gnn)

        model = Model(
            inputs=[
                A_input, 
                A_week_input, 
                A_weekend_input, 
                Temporal_input, 
                Temporal_week_input, 
                Temporal_weekend_input, 
                Distance_input, 
                Duration_input, 
                Location_time_input, 
                Location_location_input
            ], 
            outputs=[out])

        return model