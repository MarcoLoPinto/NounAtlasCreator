import torch
from torch.utils.data import DataLoader

from .Trainer import Trainer


class Trainer_nei(Trainer):

    def __init__(self):
        super().__init__()

    def init_history(self, saved_history):
        history = {}
        history['train_history'] = [] if saved_history == {} else saved_history['train_history']
        history['valid_loss_history'] = [] if saved_history == {} else saved_history['valid_loss_history']
        # history['f1_history'] = [] if saved_history == {} else saved_history['f1_history']
        return history

    def compute_forward(self, model, sample, device, optimizer = None):
        ''' must return a dictionary with "loss" key in it '''
        if optimizer is not None:
            optimizer.zero_grad()

        if sample['text_pair'][0] != '':
            predictions = model.forward(sample['text'], sample['text_pair'])
        else:
            predictions = model.forward(sample['text'])
        labels = torch.as_tensor(sample['label']).to(torch.float32)

        predictions_flattened = predictions.view(-1) # (batch , 1) -> (batch)
        labels_flattened = labels.view(-1) # (batch) -> (batch)

        predictions_flattened = predictions_flattened.to(device)
        labels_flattened = labels_flattened.to(device)

        if model.loss_fn is not None:
            sample_loss = model.compute_loss(predictions_flattened, labels_flattened)
        else:
            sample_loss = None

        if optimizer is not None:
            sample_loss.backward()
            optimizer.step()

        return {
            'labels':labels, 
            'predictions':predictions, 
            'loss':sample_loss
        }

    def compute_validation(self, model, valid_dataloader, device):
        ''' must return a dictionary with "labels", "predictions" and "loss" keys '''
        valid_loss = 0.0
        labels = []
        predictions = []

        model.eval()
        model.to(device)
        with torch.no_grad():
            for step, sample in enumerate(valid_dataloader):
                dict_out = self.compute_forward(model, sample, device, optimizer = None)

                valid_loss += dict_out['loss'].tolist() if dict_out['loss'] is not None else 0

                labels.append(dict_out['labels'])
                predictions.append(dict_out['predictions'])

        return {'labels': labels, 'predictions': predictions, 'loss': (valid_loss / len(valid_dataloader))}

    def compute_evaluations(self, labels, predictions):
        ''' must return a dictionary of results '''
        evaluations_results = {}

        return evaluations_results

    def update_history(self, history, valid_loss, evaluations_results):
        ''' must return the updated history dictionary '''

        history['valid_loss_history'].append(valid_loss)

        return history

    def print_evaluations_results(self, valid_loss, evaluations_results):
        print(f'# Validation loss => {valid_loss:0.6f} #')

    def conditions_for_saving_model(self, history, min_score):
        ''' must return True or False '''
        return (
            history['valid_loss_history'][-1] < min([99.0] + history['valid_loss_history'][:-1]) and 
            history['valid_loss_history'][-1] < min_score
        )