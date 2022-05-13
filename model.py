import os
import neuralnet as nn
from utils.common import exists
import tensorflow as tf
import numpy as np

class logger:
    def __init__(self, path, values) -> None:
        self.path = path
        self.values = values

class FSRCNN:
    def __init__(self, scale): 

        if scale not in [2, 3, 4]:
            raise ValueError("scale must be 2, 3, or 4")

        self.model = nn.FSRCNN(scale)
        self.optimizer = None
        self.loss =  None
        self.metric = None
        self.model_path = None
        self.ckpt = None
        self.ckpt_dir = None
        self.ckpt_man = None

    def setup(self, optimizer, loss, metric, model_path):
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        # @the best model weights
        self.model_path = model_path

    def load_checkpoint(self, ckpt_dir):
        self.ckpt_dir = ckpt_dir
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0), 
                                        optimizer=self.optimizer,
                                        net=self.model)
        self.ckpt_man = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=1)
        self.ckpt.restore(self.ckpt_man.latest_checkpoint)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def predict(self, lr):
        sr = self.model(lr)
        return sr

    def evaluate(self, dataset, batch_size=64):
        losses, metrics = [], []
        isEnd = False
        while isEnd == False:
            lr, hr, isEnd = dataset.get_batch(batch_size, shuffle_each_epoch=False)
            sr = self.predict(lr)
            losses.append(self.loss(hr, sr))
            metrics.append(self.metric(hr, sr))

        metric = tf.reduce_mean(metrics).numpy()
        loss = tf.reduce_mean(losses).numpy()
        return loss, metric

    def train(self, train_set, valid_set, batch_size, steps, save_every=1,
              save_best_only=False, save_log=False, log_dir=None):

        if (save_log) and (log_dir is None):
            raise ValueError("log_dir must be specified if save_log is True")
        os.makedirs(log_dir, exist_ok=True)
        dict_logger = {"loss":       logger(path=os.path.join(log_dir, "losses.npy"),      values=[]),
                       "metric":     logger(path=os.path.join(log_dir, "metrics.npy"),     values=[]),
                       "val_loss":   logger(path=os.path.join(log_dir, "val_losses.npy"),  values=[]),
                       "val_metric": logger(path=os.path.join(log_dir, "val_metrics.npy"), values=[])}
        for key in dict_logger.keys():
            path = dict_logger[key].path
            if exists(path):
                dict_logger[key].values = np.load(path).tolist()

        cur_step = self.ckpt.step.numpy()
        max_steps = steps + self.ckpt.step.numpy()

        prev_loss = np.inf
        if save_best_only and exists(self.model_path):
            self.load_weights(self.model_path)
            prev_loss, _ = self.evaluate(valid_set)
            self.load_checkpoint(self.ckpt_dir)

        loss_buffer = []
        metric_buffer = []
        while cur_step < max_steps:
            cur_step += 1
            self.ckpt.step.assign_add(1)
            lr, hr, _ = train_set.get_batch(batch_size)
            loss, metric = self.train_step(lr, hr)
            loss_buffer.append(loss)
            metric_buffer.append(metric)

            if (cur_step % save_every == 0) or (cur_step >= max_steps):
                loss = tf.reduce_mean(loss_buffer).numpy()
                metric = tf.reduce_mean(metric_buffer).numpy()
                val_loss, val_metric = self.evaluate(valid_set)
                print(f"Step {cur_step}/{max_steps}",
                      f"- loss: {loss:.7f}",
                      f"- {self.metric.__name__}: {metric:.3f}",
                      f"- val_loss: {val_loss:.7f}",
                      f"- val_{self.metric.__name__}: {val_metric:.3f}")
                
                if save_log == True:
                    dict_logger["loss"].values.append(loss)
                    dict_logger["metric"].values.append(metric)
                    dict_logger["val_loss"].values.append(val_loss)
                    dict_logger["val_metric"].values.append(val_metric)
                
                loss_buffer = []
                metric_buffer = []
                self.ckpt_man.save(checkpoint_number=0)

                if save_best_only and val_loss > prev_loss:
                    continue
                prev_loss = val_loss
                self.model.save_weights(self.model_path)
                print(f"Save model to {self.model_path}\n")
        
        if save_log == True:
            for key in dict_logger.keys():
                logger_obj = dict_logger[key]
                path = logger_obj.path
                values = np.array(logger_obj.values, dtype=np.float32)
                np.save(path, values)

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            sr = self.model(lr, training=True)
            loss = self.loss(hr, sr)
            metric = self.metric(hr, sr)
        gradient = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        return loss, metric

