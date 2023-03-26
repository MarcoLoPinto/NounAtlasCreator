import torch
from torch.utils.data import DataLoader

from .Trainer import Trainer


class Trainer_nec(Trainer):

    def __init__(self):
        super().__init__()

    def init_history(self, saved_history):
        history = {}
        history['train_history'] = [] if saved_history == {} else saved_history['train_history']
        history['valid_loss_history'] = [] if saved_history == {} else saved_history['valid_loss_history']
        history['f1_event_class_history'] = [] if saved_history == {} else saved_history['f1_event_class_history']
        return history

    def compute_forward(self, model, sample, device, optimizer = None):
        ''' must return a dictionary with "loss" key in it '''
        if optimizer is not None:
            optimizer.zero_grad()

        # inputs
        word = sample['word']
        definition = sample['definition']
        # outputs
        labels_raw = sample['label']

        predictions = model.forward(
            definition, word
        )

        predictions_raw = model.process_predictions(predictions)
        
        labels_processed = torch.as_tensor([ 
            model.hparams['predicates_to_id'][v] for v in labels_raw 
        ])

        predictions = predictions.to(device)
        labels_processed = labels_processed.to(device)

        if model.loss_fn is not None:
            # print(predictions)
            # print("-----------------")
            # print(labels_processed)
            # print(predictions.shape)
            # assert 1==2
            sample_loss = model.compute_loss(predictions, labels_processed) / 8
        else:
            sample_loss = None

        if optimizer is not None:
            sample_loss.backward()
            optimizer.step()

        return {'labels':labels_raw, 'labels_torch':labels_processed, 'predictions':predictions_raw, 'predictions_torch':predictions, 'loss':sample_loss}

    def compute_validation(self, model, valid_dataloader, device):
        ''' must return a dictionary with "labels", "predictions" and "loss" keys '''
        valid_loss = 0.0
        dict_out_all = {'labels':[],  'predictions':[]}

        model.eval()
        model.to(device)
        with torch.no_grad():
            for step, sample in enumerate(valid_dataloader):
                dict_out = self.compute_forward(model, sample, device, optimizer = None)

                valid_loss += dict_out['loss'].tolist() if dict_out['loss'] is not None else 0

                dict_out_all['labels'] += dict_out['labels']
                dict_out_all['predictions'] += dict_out['predictions']

        return {**dict_out_all, 'loss': (valid_loss / len(valid_dataloader))}

    def compute_evaluations(self, labels, predictions):
        ''' must return a dictionary of results '''
        evaluations_results = {}

        null_tag = '_' # 0 is the null tag '_'
        evaluations_results['event_class'] = Trainer_nec.evaluate_f1_event_classification(labels, predictions, null_tag)

        return evaluations_results

    def update_history(self, history, valid_loss, evaluations_results):
        ''' must return the updated history dictionary '''
        f1_event_class = evaluations_results['event_class']['f1']

        history['valid_loss_history'].append(valid_loss)
        history['f1_event_class_history'].append(f1_event_class)

        return history

    def print_evaluations_results(self, valid_loss, evaluations_results):
        f1_event_class = evaluations_results['event_class']['f1']
        print(f'# Validation loss => {valid_loss:0.6f} | f1-score: event_class = {f1_event_class:0.6f} #')

    def conditions_for_saving_model(self, history, min_score):
        ''' must return True or False '''
        return (
            history['f1_event_class_history'][-1] > max([0.0] + history['f1_event_class_history'][:-1]) and 
            history['f1_event_class_history'][-1] > min_score
        )

    ########### utils evaluations ###########

    @staticmethod
    def evaluate_f1_event_classification(labels, predictions, null_tag="_"):
        true_positives, false_positives, false_negatives = 0, 0, 0
        for sentence_id, _ in enumerate(labels):
            gold_predicates = labels[sentence_id]
            pred_predicates = predictions[sentence_id]
            for g, p in zip(gold_predicates, pred_predicates):
                if g != null_tag and p != null_tag:
                    if p == g:
                        true_positives += 1
                    else:
                        false_positives += 1
                        false_negatives += 1
                elif p != null_tag and g == null_tag:
                    false_positives += 1
                elif g != null_tag and p == null_tag:
                    false_negatives += 1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }