import torch.nn as nn
from spectre.module.unification import Unification
from spectre.module.transformers import Transformer

import torch


class BackwardChainingInference(nn.Module):
    def __init__(self, teacher_model, model_type, unification_model):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # a teacher is a pretrained model to do inference at depth=n-1
        print('Teacher model', teacher_model)
        self.teacher = Transformer(model=teacher_model, model_type=model_type).to(self.device)

        # freeze the teacher, set it in the evaluation mode
        self.teacher.freeze(self.teacher)
        self.teacher.eval()

        # a unification model will be trained to turn a query at depth=n to a query at depth=n-1
        if unification_model is None:
            self.unification = Unification(model=teacher_model, model_type=model_type)
        else:
            self.unification = Unification(model=unification_model, model_type=model_type)

    @staticmethod
    def freeze(m):
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, contexts, questions, answers):
        context_question_embeds, masks, types = self.unification(contexts, questions)
        loss, logits = self.teacher(context_question_embeds, masks, types, answers)
        return loss, logits, context_question_embeds, masks, types
