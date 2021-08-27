import torch.nn as nn
from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel
from spectre.module.utils.utils import padding_sentences, padding_sentence_embeddings

import torch


class Unification(nn.Module):
    def __init__(self, model_type, model):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        if self.model_type == 'bert':
            print('Unification model:', model)
            self.model = BertModel.from_pretrained(model).to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(model)
        else:
            print('Unification model:', model)
            self.model = RobertaModel.from_pretrained(model).to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(model)
        self.freeze(self.model.embeddings.word_embeddings)

    @staticmethod
    def freeze(m):
        for p in m.parameters():
            p.requires_grad = False

    def data_prep(self, contexts, questions):
        b_size = len(questions)
        context_question_embeds = []
        context_embeds = []
        token_types = []
        for i in range(b_size):
            context_ids = torch.tensor(self.tokenizer.encode(contexts[i], add_special_tokens=True)).to(self.device)
            question_ids = torch.tensor(self.tokenizer.encode(questions[i], add_special_tokens=True)).to(self.device)
            question_embed = self.model.embeddings.word_embeddings(question_ids)
            context_embed = self.model.embeddings.word_embeddings(context_ids)
            token_type = torch.tensor([0]*len(context_embed) + [1]*len(question_embed))
            token_types.append(token_type)
            context_question_embeds.append(torch.cat([context_embed, question_embed]))
            context_embeds.append(context_embed)

        context_question_embed, masks = padding_sentence_embeddings(context_question_embeds)
        types, _ = padding_sentences(token_types)
        return context_question_embed, context_embeds, masks, types

    @staticmethod
    def replace_context(outputs, context_embeds):
        for i in range(len(context_embeds)):
            outputs[i, :len(context_embeds[i]), :] = context_embeds[i]
        return outputs

    def forward(self, contexts, questions):
        context_question_embeds, context_embeds, masks, types = self.data_prep(contexts, questions)
        if self.model_type == 'bert':
            outputs, _ = self.model(
                                 inputs_embeds=context_question_embeds.to(self.device),
                                 attention_mask=masks.to(self.device),
                                 token_type_ids=types.to(self.device))
        else:
            outputs, _ = self.model(
                inputs_embeds=context_question_embeds.to(self.device),
                attention_mask=masks.to(self.device))

        # keep original context embeds, only care about the question output
        outputs = self.replace_context(outputs, context_embeds)

        # set the padding to zeros
        x = masks == 0
        outputs[x] = torch.zeros_like(outputs[x])
        return outputs, masks, types
