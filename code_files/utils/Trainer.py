import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Trainer():

    def __init__(self):
        pass

    def train(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader = None,
        epochs: int = 5,
        verbose: bool = True,
        save_best = True,
        save_path_name = None,
        min_score = 0.5,
        saved_history = {},
        device = 'cpu'
    ):  
        """Training and evaluation function in order to save the best model

        Args:
            model (any): the ML model
            optimizer (torch.optim.Optimizer): the torch.optim.Optimizer used 
            train_dataloader (DataLoader): the train data created with torch.utils.data.Dataloader
            valid_dataloader (DataLoader, optional): the dev data created with torch.utils.data.Dataloader. Defaults to None.
            epochs (int, optional): number of maximum epochs. Defaults to 5.
            verbose (bool, optional): if True, then each epoch will print the training loss, the validation loss and the f1-score. Defaults to True.
            save_best (bool, optional): if True, then the best model that surpasses min_score will be saved. Defaults to True.
            save_path_name (str, optional): path and name for the best model to be saved. Defaults to None.
            min_score (float, optional): minimum score acceptable in order to be saved. Defaults to 0.5.
            saved_history (dict, optional): saved history dictionary from another session. Defaults to {}.
            device (str, optional): if we are using cpu or gpu. Defaults to 'cpu'.

        Returns:
            a dictionary of histories
        """

        history = self.init_history(saved_history) # override

        model.to(device)

        for epoch in range(epochs):
            losses = []
            
            model.train()

            # batches of the training set
            for step, sample in enumerate(train_dataloader):
                dict_out = self.compute_forward(model, sample, device, optimizer = optimizer) # override
                losses.append(dict_out['loss'].item())

            mean_loss = sum(losses) / len(losses)
            history['train_history'].append(mean_loss)
            
            if verbose or epoch == epochs - 1:
                print(f'Epoch {epoch:3d} => avg_loss: {mean_loss:0.6f}')
            
            if valid_dataloader is not None:

                valid_out = self.compute_validation(model, valid_dataloader, device) # override

                evaluations_results = self.compute_evaluations(valid_out['labels'], valid_out['predictions']) # override

                history = self.update_history(history, valid_out['loss'], evaluations_results) # override

                if verbose:
                    self.print_evaluations_results(valid_out['loss'], evaluations_results) # override

                # saving...

                if save_best and save_path_name is not None:
                    if self.conditions_for_saving_model(history, min_score): # override
                        print(f'----- Best value obtained, saving model -----')
                        model.save_weights(save_path_name)
                    
        return history


    def init_history(self, saved_history):
        ''' must return the initialized history dictionary '''
        raise NotImplementedError

    def compute_forward(self, model, sample, device, optimizer = None):
        ''' must return a dictionary with "loss" key in it '''
        raise NotImplementedError

    def compute_validation(self, model, valid_dataloader, device):
        ''' must return a dictionary with "labels", "predictions" and "loss" keys '''
        raise NotImplementedError

    def compute_evaluations(self, labels, predictions):
        ''' must return a dictionary of results '''
        raise NotImplementedError

    def update_history(self, history, valid_loss, evaluations_results):
        ''' must return the updated history dictionary '''
        raise NotImplementedError

    def print_evaluations_results(self, valid_loss, evaluations_results):
        print('Not implemented.')
        raise NotImplementedError

    def conditions_for_saving_model(self, history, min_score):
        ''' must return True or False '''
        raise NotImplementedError

    @staticmethod
    def display_history(dict_history):
        plt.figure(figsize=(8,8))
        for name, hist in dict_history.items():
            plt.plot([i for i in range(len(hist))], hist, label=name)
        plt.xlabel('epochs')
        plt.ylabel('value')
        plt.title('Model learning')
        plt.legend()
        plt.show()