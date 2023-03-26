import torch
import torch.nn as nn
import numpy as np
import typing

from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

class NEClassifier(nn.Module):
    def __init__(   self, 
                    hparams = {}, 
                    loss_fn = None,
                    fine_tune_transformer = False):
        """Nominal Event Classifier model

        Args:
            hparams (any, optional): Parameters necessary to initialize the model. It can be either a dictionary or the path string for the file. Defaults to {}.
            loss_fn (any, optional): Loss function. Defaults to None.
            fine_tune_transformer (bool, optional): If the transformer needs to be fine-tuned. Defaults to False.
        """
                    
        super().__init__()

        self.hparams = hparams if type(hparams) != str else self._load_hparams(hparams)

        self.tokenizer = AutoTokenizer.from_pretrained(hparams['transformer_name'])

        self.n_labels = int(hparams['n_predicates_labels'])
        
        # 1) Embedding
        self.transformer_model = AutoModel.from_pretrained(
            hparams['transformer_name'], output_hidden_states=True
        )
        if not fine_tune_transformer:
            for param in self.transformer_model.parameters():
                param.requires_grad = fine_tune_transformer

        transformer_out_dim = self.transformer_model.config.hidden_size

        self.dropout = nn.Dropout(0.2)

        # 2) Classifier

        self.classifier = nn.Linear(transformer_out_dim, self.n_labels)

        # Loss function:
        self.loss_fn = loss_fn
    
    def forward(
        self, 
        text: typing.Union[str, typing.List[str], typing.List[typing.List[str]]],
        text_pair: typing.Union[str, typing.List[str], typing.List[typing.List[str]], None] = None,
        is_split_into_words: bool = False,
    ):

        device = self.get_device()

        # 1) Embedding

        batch_encoding = self.tokenizer(
            text, text_pair,
            return_tensors="pt",
            padding=True,
            is_split_into_words=is_split_into_words,
            # truncation=True # Warning!
        ).to(device)

        transformer_outs = self.transformer_model(**batch_encoding)

        # number of hidden states to consider from the transformer
        n_transformer_hidden_states = 1
        # summing all the considered dimensions
        transformer_out = torch.stack(
            transformer_outs.hidden_states[-n_transformer_hidden_states:],
            dim=0).sum(dim=0)

        cls_token = transformer_out[:,0,:] # (batch, sentence_len, word_emb_dim) -> (batch, word_emb_dim)

        batch_sentence_words = self.dropout(cls_token) # -> (batch, word_emb_dim)

        # 2) Classifier

        predictions = self.classifier(batch_sentence_words)

        return predictions # predictions = (batch, n_lables)

    def process_predictions(self, predictions):
        indices = self.get_indices(predictions).cpu().detach().numpy()
        predictions_processed = [ self.hparams['id_to_predicates'][v] for v in indices ]
        return predictions_processed

    def predict(
        self,
        text: typing.Union[typing.List[str], typing.List[typing.List[str]]],
        text_pair: typing.Union[typing.List[str], typing.List[typing.List[str]], None] = None,
        is_split_into_words: bool = False
    ):
        """Predict function to use after training/loading the model

        Args:
            text (typing.Union[typing.List[str], typing.List[typing.List[str]]]): The tokenized sentence (or list of sentences).
            text_pair (typing.Union[typing.List[str], typing.List[typing.List[str]]]): The predicate word (or list of predicates). Defaults to None (do not change it unless trained otherwise).
            is_split_into_words (bool, optional): If it's split into words. Defaults to True (do not change it unless trained otherwise).

        Returns:
            typing.Union[str, typing.List[str]]: The predicted predicate frame name for the sentence (or list of frame names).
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(text,text_pair,is_split_into_words)
            return self.process_predictions(predictions)

    def compute_loss(self, x, y_true):
        """computes the loss for the net

        Args:
            x (torch.Tensor): The predictions
            y_true (torch.Tensor): The true labels

        Returns:
            any: the loss
        """
        if self.loss_fn is None:
            return None
        return self.loss_fn(x, y_true)

    def get_indices(self, torch_outputs):
        """

        Args:
            torch_outputs (torch.Tensor): a Tensor with shape (batch_size, sentence_len, label_vocab_size) containing the logits outputed by the neural network (if batch_first = True)
        
        Returns:
            The method returns a (batch_size, sentence_len) shaped tensor (if batch_first = True)
        """
        max_indices = torch.argmax(torch_outputs, -1) # resulting shape = (batch_size, sentence_len)
        return max_indices
    
    def load_weights(self, path, strict = True):
        """load the weights of the model

        Args:
            path (str): path to the saved weights
            strict (bool, optional): Strict parameter for the torch.load() function. Defaults to True.
        """
        self.load_state_dict(torch.load(path, map_location=next(self.parameters()).device), strict=strict)
        self.eval()
    
    def save_weights(self, path):
        """save the weights of the model

        Args:
            path (str): path to save the weights
        """
        torch.save(self.state_dict(), path)

    def _load_hparams(self, hparams):
        """loads the hparams from the file

        Args:
            hparams (str): the hparams path

        Returns:
            dict: the loaded hparams
        """
        return np.load(hparams, allow_pickle=True).tolist()

    def get_device(self):
        """get the device where the model is

        Returns:
            str: the device ('cpu' or 'cuda')
        """
        return next(self.parameters()).device
