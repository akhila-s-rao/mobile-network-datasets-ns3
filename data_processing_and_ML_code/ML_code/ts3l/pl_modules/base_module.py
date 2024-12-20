from typing import Dict, Any, List, Type
from ts3l.utils import RegressionMetric, ClassificationMetric

from abc import ABC, abstractmethod

import torch
from torch import nn

import pytorch_lightning as pl

from dataclasses import asdict
from ts3l.utils import BaseConfig
from ts3l.models.common import initialize_weights

    
class TS3LLightining(ABC, pl.LightningModule):
    """The pytorch lightning module of TabularS3L
    """
    def __init__(self, config: BaseConfig) -> None:
        """Initialize the pytorch lightining module of TabularS3L

        Args:
            config (BaseConfig): The configuration of TS3LLightining.
        """
        super(TS3LLightining, self).__init__()
        
        _config = asdict(config)
        
        self.random_seed = _config["random_seed"]
        del _config["random_seed"]
        
        pl.seed_everything(self.random_seed)
        
        self.optim = getattr(torch.optim, _config["optim"])
        del _config["optim"]
        self.optim_hparams = _config["optim_hparams"]
        del _config["optim_hparams"]
        
        self.sched = getattr(torch.optim.lr_scheduler, _config["scheduler"]) if _config["scheduler"] is not None else None
        del _config["scheduler"]
        self.scheduler_hparams = _config["scheduler_hparams"]
        del _config["scheduler_hparams"]
        
        self.task_loss_fn = getattr(torch.nn, _config["loss_fn"])(**_config["loss_hparams"])
        del _config["loss_fn"]
        del _config["loss_hparams"]
        
        self.__configure_metric(_config["task"], _config["metric"], _config["metric_hparams"])
        del _config["task"]
        del _config["metric"]
        del _config["metric_hparams"]
        
        self._initialize(_config)
        
        self.set_first_phase()

        self.first_phase_step_outputs: List[Dict[str, Any]] = []
        self.second_phase_step_outputs: List[Dict[str, Any]] = []
        
        self.save_hyperparameters()

        # Akhila 15 oct 
        # Initialize lists to store losses
        self.first_phase_train_loss = []
        self.first_phase_val_loss = []
        self.second_phase_train_loss = []
        self.second_phase_val_loss = []
        self.second_phase_val_metric = []
    
    @abstractmethod
    def _initialize(self, config: Dict[str, Any]) -> None:
        pass
    
    def _init_model(self, model_class: Type[nn.Module], config: Dict[str, Any]) -> None:
        initialization = config["initialization"]
        del config["initialization"]
        
        self.model = model_class(**config)
        initialize_weights(self.model, initialization)
        
    def __configure_metric(self, task, metric, metric_hparams):
        
        if task == "regression":
            self.metric = RegressionMetric(metric, metric_hparams)
        else:
            self.metric = ClassificationMetric(metric, metric_hparams)
            
    def configure_optimizers(self):
        """Configure the optimizer
        """
        self.optimizer = self.optim(self.parameters(), **self.optim_hparams)
        if self.sched is None:
            return [self.optimizer]
        self.scheduler = self.sched(self.optimizer, **self.scheduler_hparams)
        return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step'} ]

    def set_first_phase(self) -> None:
        """Set the module to pretraining
        """
        self.model.set_first_phase()
        self.training_step = self._first_phase_step # type: ignore
        self.on_validation_start = self._on_first_phase_validation_start # type: ignore
        self.validation_step = self._first_phase_step # type: ignore
        self.on_validation_epoch_end = self._first_phase_validation_epoch_end # type: ignore

    # Akhila
    def get_first_phase_output(self, dataloader) -> torch.Tensor:
        # Get the output of the enmedding layers 
        print('Inside get_first_phase_output function')
        print(self.model.encoder)
        print(dataloader)
        # Turn off gradient calculations since this is inference
        self.model.encoder.eval()
        
        outputs = []
        with torch.no_grad():
            for batch in dataloader:
                print('input batch: ', batch)
                # Assuming the input data is the first element in the batch
                input_data = batch[0]
                
                # Feed the data into the encoder
                encoded_output = self.model.encoder(input_data)
                print('output embedding: ', encoded_output)
                # Append the encoded output to the list
                outputs.append(encoded_output)
        
        # Concatenate all outputs into a single tensor
        encoded_outputs_tensor = torch.cat(outputs, dim=0)
        
        # Now `encoded_outputs_tensor` contains the encoded representation of your entire dataset
        print(encoded_outputs_tensor.shape)

        return encoded_outputs_tensor
        
    def set_second_phase(self, freeze_encoder: bool, pred_head_size: int = 1) -> None:
        """Set the module to fine-tuning
        
        Args:
            freeze_encoder (bool): If True, the encoder will be frozen during fine-tuning. Otherwise, the encoder will be trainable.
        """
        self.model.set_second_phase(freeze_encoder, pred_head_size)
        self.training_step = self._second_phase_step # type: ignore
        self.on_validation_start = self._on_second_phase_validation_start # type: ignore
        self.validation_step = self._second_phase_step # type: ignore
        self.on_validation_epoch_end = self._second_phase_validation_epoch_end # type: ignore

    def forward(self,
                batch:Dict[str, Any]
    ) -> torch.Tensor:
        """Do forward pass for given input

        Args:
            batch (Dict[str, Any]): The input batch

        Returns:
            torch.Tensor: The output of forward pass
        """
        return self.model(batch)
    

    @abstractmethod
    def _get_first_phase_loss(self, batch: Any) -> torch.Tensor:
        """Calculate the first phase loss

        Args:
            batch (Any): The input batch

        Returns:
            torch.Tensor: The final loss of first phase step
        """
        pass
    
    def _first_phase_step(self,
                      batch: Any,
                      batch_idx: int
    ) -> Dict[str, Any]:
        """The first phase step of TabularS3L

        Args:
            batch (Any): The input batch
            batch_idx (int): Only for compatibility

        Returns:
            Dict[str, Any]: The loss of the first phase step
        """

        loss = self._get_first_phase_loss(batch)
        self.first_phase_step_outputs.append({
            "loss" : loss
        })
        return {
            "loss" : loss
        }

    def _on_first_phase_validation_start(self):
        """Log the training loss of the first_phase
        """
        if len(self.first_phase_step_outputs) > 0:
            train_loss = torch.Tensor([out["loss"] for out in self.first_phase_step_outputs]).cpu().mean()
            
            self.log("train_loss", train_loss, prog_bar = True)
            # Akhila 15 oct
            # Trying to save this in the module itself to access later
            self.first_phase_train_loss.append(train_loss)
            
            self.first_phase_step_outputs = []    
        return super().on_validation_start() 
    
    def _first_phase_validation_epoch_end(self) -> None:
        """Log the validation loss of the first phase
        """
        val_loss = torch.Tensor([out["loss"] for out in self.first_phase_step_outputs]).cpu().mean()

        self.log("val_loss", val_loss, prog_bar = True)
        # Akhila 15 oct
        # Trying to save this in the module itself to access later
        self.first_phase_val_loss.append(val_loss)
        
        self.first_phase_step_outputs = []
        return super().on_validation_epoch_end()

    @abstractmethod
    def _get_second_phase_loss(self, batch: Any):
        """Calculate the second phase loss

        Args:
            batch (Any): The input batch

        Returns:
            torch.FloatTensor: The final loss of second phase step
            torch.Tensor: The label of the labeled data
            torch.Tensor: The predicted label of the labeled data
        """
        pass
        
    
    def _second_phase_step(self,
                      batch: Any,
                      batch_idx: int = 0
    ) -> Dict[str, Any]:
        """The second phase step of TabularS3L

        Args:
            batch (Any): The input batch
            batch_idx (int): Only for compatibility

        Returns:
            Dict[str, Any]: The loss of the second phase step
        """
        loss, y, y_hat = self._get_second_phase_loss(batch)

        self.second_phase_step_outputs.append(
            {
            "loss" : loss,
            "y" : y,
            "y_hat" : y_hat
        }
        )
        return {
            "loss" : loss
        }
    
    def _on_second_phase_validation_start(self):
        """Log the training loss and the performance of the second phase
        """
        if len(self.second_phase_step_outputs) > 0:
            train_loss = torch.Tensor([out["loss"] for out in self.second_phase_step_outputs]).detach().mean()
            y = torch.cat([out["y"] for out in self.second_phase_step_outputs if out["y"].numel() != 1]).detach().cpu()
            y_hat = torch.cat([out["y_hat"] for out in self.second_phase_step_outputs if out["y_hat"].numel() != 1]).detach().cpu()
            train_score = self.metric(y_hat, y)
            
            self.log("train_loss", train_loss, prog_bar = True)
            
            # Akhila 15 oct
            # Trying to save this in the module itself to access later
            self.second_phase_train_loss.append(train_loss)
            
            # Akhila did this to reduce the values seen on screen 
            #self.log("train_" + self.metric.__name__, train_score, prog_bar = True)
            self.second_phase_step_outputs = []   
            
        return super().on_validation_start()
    
    def _second_phase_validation_epoch_end(self) -> None:
        """Log the validation loss and the performance of the second phase
        """
        val_loss = torch.Tensor([out["loss"] for out in self.second_phase_step_outputs]).mean()

        
        y = torch.cat([out["y"].cpu() for out in self.second_phase_step_outputs if out["y"].numel() != 1])
        y_hat = torch.cat([out["y_hat"].cpu() for out in self.second_phase_step_outputs if out["y_hat"].numel() != 1])
        val_score = self.metric(y_hat, y)
        
        # Akhila 15 oct
        # Trying to save this in the module itself to access later
        self.second_phase_val_loss.append(val_loss)
        self.second_phase_val_metric.append(val_score)
        
        
        self.log("val_" + self.metric.__name__, val_score, prog_bar = True)
        # Akhila did this to reduce the values seen on screen 
        #self.log("val_loss", val_loss, prog_bar = True)
        self.second_phase_step_outputs = []      
        return super().on_validation_epoch_end()
    
    @abstractmethod
    def predict_step(self, batch: Any, batch_idx: int
    ) -> Any:
        """The perdict step

        Args:
            batch (Any): The input batch
            batch_idx (int): Only for compatibility

        Returns:
            Any: The predicted output and optional additional information.
        """
        pass