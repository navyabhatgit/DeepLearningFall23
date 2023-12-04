import pytorch_lightning as pl
import torch

class MultiClassLightningModule(pl.LightningModule):
    def __init__(self, model, optimizer_cls, loss_fn, metric_cls, num_classes, learning_rate, optimizer_params=None,
                 scheduler_cls=None, scheduler_params=None, scheduler_options=None, log_every_n_steps=50,
                 log_test_metrics=True, display_metrics=True,):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.train_metric = metric_cls(task="multiclass", num_classes=self.num_classes)
        self.val_metric = metric_cls(task="multiclass", num_classes=self.num_classes)
        self.test_metric = metric_cls(task="multiclass", num_classes=self.num_classes)
        self.log_every_n_steps = log_every_n_steps
        self.log_test_metrics = log_test_metrics
        self.optimizer_cls = optimizer_cls
        self.display_metrics = display_metrics
        self.optimizer_params = optimizer_params if optimizer_params else {}
        self.scheduler_cls = scheduler_cls
        self.scheduler_params = scheduler_params if scheduler_params else {}
        self.scheduler_options = scheduler_options if scheduler_options else {}
        self.save_hyperparameters(ignore=['model', 'loss_fn'])
        

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        inputs, labels = batch
        output = self(inputs)
        loss = self.loss_fn(output, labels)
        predicted_labels = torch.argmax(output, dim=1)
        return loss, labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self._shared_step(batch)
        if batch_idx % self.log_every_n_steps == 0:
            self.log("train_loss_step", loss, on_step=True, on_epoch=False)
        self.train_metric(predicted_labels, labels)
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_metric", self.train_metric, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def on_train_epoch_end(self):
        if self.display_metrics:
            metrics = self.trainer.callback_metrics
            print(f"Train_Loss: {metrics['train_loss_epoch']:.2f}, Train_Metric: {metrics['train_metric']:.2f}")
    
    def validation_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self._shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        self.val_metric(predicted_labels, labels)
        self.log("val_metric", self.val_metric, prog_bar=True, on_epoch=True, on_step=False)
    
    def on_validation_epoch_end(self):
        if self.display_metrics:
            metrics = self.trainer.callback_metrics
            epoch_num = self.current_epoch
            print(f"Epoch {epoch_num + 1}: Val_Loss: {metrics['val_loss']:.2f}, Val_Metric: {metrics['val_metric']:.2f}", end=" | ", flush=True)

    def test_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self._shared_step(batch)
        self.test_metric(predicted_labels, labels)
        if self.log_test_metrics:
            self.log("test_metric", self.test_metric)
    
    def on_test_epoch_end(self):
        if not self.log_test_metrics:
            computed_test_metric = self.test_metric.compute()
            print(f"Test Metric: {computed_test_metric:.2f}")
                
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if len(batch) == 2:
            _, labels, predicted_labels = self._shared_step(batch)
        else:
            inputs = batch
            output = self(inputs)
            predicted_labels = torch.argmax(output, dim=1)
        
        return predicted_labels
    
    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        
        if self.scheduler_cls is not None:
            scheduler = self.scheduler_cls(optimizer, **self.scheduler_params)
            
            scheduler_dict = {
                'scheduler': scheduler,
                **self.scheduler_options  # Spread the scheduler options here
            }
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler_dict
            }
        
        return optimizer