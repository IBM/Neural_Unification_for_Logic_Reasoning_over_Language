import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, RobertaForSequenceClassification, \
    RobertaTokenizer
from spectre.module.utils.utils import padding_sentences, padding_sentence_embeddings
from spectre.module.utils.utils import weight_stats

import torch


class Transformer(nn.Module):
    def __init__(self, model, model_type, pretrained_model=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        if self.model_type == 'bert':
            self.model = BertForSequenceClassification.from_pretrained(model).to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(model)
            if pretrained_model is not None:
                print("Load base model from provided path:", pretrained_model)
                print(weight_stats(self.model.bert))
                self.model.bert = self.model.bert.from_pretrained(pretrained_model)
                self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
            self.base_model = self.model.bert
            print(weight_stats(self.base_model))
        else:
            print('Using model:', model)
            self.model = RobertaForSequenceClassification.from_pretrained(model).to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(model)
            if pretrained_model is not None:
                print("Load base model from provided path:", pretrained_model)
                print(weight_stats(self.model.roberta))
                self.model.roberta = self.model.roberta.from_pretrained(pretrained_model)
                self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
            self.base_model = self.model.roberta
            print(weight_stats(self.base_model))

    @staticmethod
    def freeze(m):
        for p in m.parameters():
            p.requires_grad = False

    def data_prep(self, contexts, questions):
        b_size = len(questions)
        tokens = []
        token_types = []
        for i in range(b_size):
            context_ids = torch.tensor(self.tokenizer.encode(contexts[i], add_special_tokens=True)).to(self.device)
            question_ids = torch.tensor(self.tokenizer.encode(questions[i], add_special_tokens=True)).to(self.device)
            question_embed = self.base_model.embeddings.word_embeddings(question_ids)
            context_embed = self.base_model.embeddings.word_embeddings(context_ids)
            token_type = torch.tensor([0] * len(context_embed) + [1] * len(question_embed))
            token_types.append(token_type)
            tokens.append(torch.cat([context_embed, question_embed]))

        embeds, masks = padding_sentence_embeddings(tokens)
        types, _ = padding_sentences(token_types)
        return embeds, masks, types

    def forward(self, embeds, masks, types, answers):
        if self.model_type == 'bert':
            outputs = self.model(
                inputs_embeds=embeds.to(self.device),
                attention_mask=masks.to(self.device),
                token_type_ids=types.to(self.device),
                labels=answers.to(self.device)
            )
        else:
            outputs = self.model(
                inputs_embeds=embeds.to(self.device),
                attention_mask=masks.to(self.device),
                labels=answers.to(self.device)
            )

        loss, logits = outputs[:2]
        return loss, logits
