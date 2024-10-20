#=============================================
# DANN 
#=============================================

from helper_functions import *

# Define the domain classifier with a gradient reversal layer
def gradient_reversal(x):
    return -x

# Define loss functions
def custom_regression_loss(y_true, y_pred):
    # Define your regression loss function here (e.g., mean squared error)
    return K.mean(K.square(y_true - y_pred))

def domain_adversarial_loss(y_true, y_pred):
    # Binary cross-entropy loss for domain classification
    return K.mean(K.binary_crossentropy(y_true, y_pred))


def dann(input_dim, X_train, y_train_regression, domain_labels):
    # Define the input shape
    input_shape = (input_dim,)

    # Define the feature extractor shared by the regressor and domain classifier
    input_layer = Input(shape=input_shape)
    shared_feature_extractor = Dense(64, activation='relu')(input_layer)  # Adjust the architecture as needed

    # Define the regressor for the regression task
    regressor = Dense(1, activation='linear')(shared_feature_extractor)  # Linear activation for regression

    domain_classifier = Dense(1, activation='sigmoid')(Lambda(gradient_reversal)(shared_feature_extractor))

    # Create the DANN model
    dann_model = Model(inputs=input_layer, outputs=[regressor, domain_classifier])

    # Compile the model
    dann_model.compile(
        optimizer=Adam(lr=IN_PARAM['learning_rate']),
        loss={'regressor': custom_regression_loss, 'domain_classifier': domain_adversarial_loss},
        loss_weights={'regressor': 1.0, 'domain_classifier': 1.0}  # Adjust the weights as needed
    )

    # Train the model
    dann_model.fit(
        x_train,
        {'regressor': y_train_regression, 'domain_classifier': domain_labels},  # domain_labels are 0 for source, 1 for target
        epochs=IN_PARAM['epochs'],
        batch_size=IN_PARAM['batch_size']
    )

    # Make predictions using the regressor
    predictions = dann_model.predict(x_test)[0]  # Index 0 corresponds to the regressor output
