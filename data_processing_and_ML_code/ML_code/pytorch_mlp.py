from helper_functions import *

class MLPRegressor(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, use_batchnorm=False, 
                 dropout_rate=0.1, learning_rate=0.0001, loss_fn_str='MSELoss', bin_edges=None, bin_weights=None):
        super(MLPRegressor, self).__init__()
        self.save_hyperparameters()  # Saves all the arguments passed to __init__
        
        self.learning_rate = learning_rate
        # Model layers
        layers = []
        # Input to hidden layer
        for i in range(depth - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            # ReLU activation
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        
        # last hidden layer
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        # additional head layer 
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Define the model as a sequential container
        self.model = nn.Sequential(*layers)

        print('loss_fn_str', loss_fn_str)
        # loss function to use
        if loss_fn_str == 'MAPELoss':
            self.loss_fn = MAPELoss()
        elif loss_fn_str == 'WeightedL1Loss':
            self.loss_fn = WeightedL1Loss(bin_edges, bin_weights)
        else:
            self.loss_fn = getattr(nn, loss_fn_str)()

        self.val_metric_fn = R2Score()
        print('USING LOSS FN: ', self.loss_fn)
        
        # Initialize lists to store losses
        self.train_losses = []
        self.val_losses = []
        #self.val_mapes = []

        # Initialize variables to accumulate losses within an epoch
        self.train_loss_epoch = []
        self.val_loss_epoch = []
        #self.val_mape_epoch = []

    
    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        #loss = F.mse_loss(y_hat, y)
        loss = self.loss_fn(y_hat, y)
        self.train_loss_epoch.append(loss.item())
        self.log('train_losses', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        #val_loss = self.loss_fn(y_hat, y)
        #val_mape = torch.mean(torch.abs((y_hat - y) / y)) * 100
        val_loss = self.val_metric_fn(y_hat, y)
        self.val_loss_epoch.append(val_loss.item())
        #self.val_r2_epoch.append(val_r2.item())
        self.log('val_losses', val_loss, on_step=False, on_epoch=True)
        #self.log('val_r2', val_r2, on_step=False, on_epoch=True)
        return val_loss
        
    def on_train_epoch_end(self):
        # Calculate and store the average training loss for the epoch
        avg_train_loss = np.mean(self.train_loss_epoch)
        self.train_losses.append(avg_train_loss)
        self.train_loss_epoch = []  # Reset for the next epoch

    def on_validation_epoch_end(self):
        # Calculate and store the average validation loss for the epoch
        avg_val_loss = np.mean(self.val_loss_epoch)
        self.val_losses.append(avg_val_loss)
        self.val_loss_epoch = []  # Reset for the next epoch
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer