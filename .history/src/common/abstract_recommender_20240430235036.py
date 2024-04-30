import os
import numpy as np
import torch
import torch.nn as nn


class AbstractRecommender(nn.Module):
    r"""Base class for all models
    """
    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r"""full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """
    def __init__(self, config, dataloader):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataloader.dataset.get_user_num()
        self.n_items = dataloader.dataset.get_item_num()

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.device = config['device']
        self.view_num = 4
        self.activation = nn.LeakyReLU()
        self.temperature = config['temperature']
        self.text_type = config['text_type']

        # load encoded features here
        self.v_feat, self.t_feat = None, None
        if not config['end2end'] and config['is_multimodal_model']:
            dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
            v_feat_file_path = os.path.join(dataset_path, config['vision_feature_file'])
            t_feat_file_path = os.path.join(dataset_path, config['text_feature_file'])
            if os.path.isfile(v_feat_file_path):
                self.v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)
                
            if os.path.isfile(t_feat_file_path):
                self.t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)            
                if self.text_type == 'MLP':
                    self.mlp = nn.Linear(self.t_feat.shape[0]*self.t_feat.shape[2], self.t_feat.shape[2], device=self.device)
                    self.t_feat = torch.cat([i for i in self.t_feat], dim=1)
                    self.t_feat = self.activation(self.mlp(self.t_feat))
                if self.text_type == 'Attention':
                    attention = torch.stack([torch.cosine_similarity(text_feat, self.v_feat, dim=-1)/self.temperature for text_feat in self.t_feat], dim=-1).softmax(dim=-1)
                    self.t_feat = (self.t_feat.transpose(0,1) * attention.unsqueeze(-1)).sum(dim=1)
                if self.text_type == 'Concat':
                    self.t_feat = torch.cat([i for i in self.t_feat], dim=1)
                if self.text_type == 'Mean':
                    self.t_feat = torch.mean(self.t_feat[:3], dim=0)
                if self.text_type == 'Sum':
                    self.t_feat = torch.sum(self.t_feat, dim=0)

            assert self.v_feat is not None or self.t_feat is not None, 'Features all NONE'        

