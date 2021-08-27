import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from spectre.module.transformers import Transformer
from .datasets import RuleReasoningDataset, BirdElectricityDataset, NatLangDataset, CWA, RuleReasoningDatasetOnlyCWA, \
    RuleReasoningDatasetOnlyNotCWA
from .module.backward_chaining_inference import BackwardChainingInference
from .module.utils.utils import decode_nearest
import argparse
import ast


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ruletaker_predictions(rule_taker, question, context, answer):
    embeds, masks, types = rule_taker.data_prep(context, question)
    loss, output = rule_taker(embeds, masks, types, answer)
    return output.detach()


def tune(batch_size, root, depth, select, teacher, unification, ruletaker, name_data, cwa):
    tune_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_tunes = len(tune_weights)
    tune_data = 'dev'
    if name_data == 'synthetic':
        if cwa == CWA.ALL:
            data_set = RuleReasoningDataset(root, depth, tune_data, select)
        elif cwa == CWA.CWA:
            data_set = RuleReasoningDatasetOnlyCWA(root, depth, tune_data, select)
        elif cwa == CWA.NOT_CWA:
            data_set = RuleReasoningDatasetOnlyNotCWA(root, depth, tune_data, select)
    elif name_data == 'nature':
        data_set = NatLangDataset(root, tune_data, select, cwa=cwa)
    else:
        data_set = BirdElectricityDataset(root, tune_data, select, name_data, cwa=cwa)
    valid_set = DataLoader(
        data_set, batch_size=batch_size, num_workers=4
    )
    dataset = iter(valid_set)
    correct = [0.0]*n_tunes
    n = 0.0

    print('Load unification model from', unification)
    net_backward.load_state_dict(torch.load(unification))

    print('Load teacher model from', teacher)
    net_backward.teacher.load_state_dict(torch.load(teacher))
    net_backward.eval()

    print('Load ruletaker model from', ruletaker)
    net_ruletaker.load_state_dict(torch.load(ruletaker))
    net_ruletaker.eval()

    with torch.no_grad():
        i = 0
        for context, question,  answer in tqdm(dataset):
            loss, output, embeds, masks, types = net_backward(context, question, answer)
            rt_pred = ruletaker_predictions(net_ruletaker, question, context, answer)
            for j in range(n_tunes):
                preds = tune_weights[j]*rt_pred + (1-tune_weights[j])*output.detach()
                correct[j] += (preds.argmax(1) == answer.to(device)).sum()
            n += len(output)
            if i < 10:
                print(decode_nearest(embeds[0], net_backward.teacher.base_model.embeddings.word_embeddings.weight,
                                     net_backward.teacher.tokenizer, masks[0], types[0]), '[QUES]', question[0], answer[0].item(), output.detach().argmax(1)[0].item())
            i += 1

    max_correct = 0.0
    max_weight = 0.0
    for j in range(n_tunes):
        print(
            'Average Accuracy: {:.5f}'.format(
                correct[j]/n
            )
        )
        if correct[j]/n > max_correct:
            max_correct = correct[j]/n
            max_weight = tune_weights[j]
    print("Max accuracy:", max_correct, "Achieved with weight", max_weight)
    return max_weight


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a tuned')
    parser.add_argument(
        '-r', '--root', required=True, type=str, help='Root directory with the training data')
    parser.add_argument(
        '-t', '--teacher', type=str, required=True, help='Location of the teacher model')
    parser.add_argument(
        '-u', '--unification', type=str, required=True, help='Location of the unification model')
    parser.add_argument(
        '-rt', '--ruletaker', type=str, required=True, help='Location of the ruletaker model')
    parser.add_argument(
        '-d', '--depths', type=str, default='[3]', help='Reasoning depths')
    parser.add_argument(
        '-l', '--lr',  type=float, default=10e-3, help='Learning rate')
    parser.add_argument(
        '-e', '--epochs', type=int, default=20, help='The number of epochs')
    parser.add_argument(
        '-b', '--batch', type=int, default=64, help='Mini-batch size')
    parser.add_argument(
        '-s', '--select', type=str, default='[0,1,2,3,4,5]',
        help='Whether only consider only questions with the given selected depths')
    parser.add_argument(
        '-n', '--name_data', type=str, default='synthetic',
        help='Data set name: synthetic, nature, electricity')
    parser.add_argument(
        '-m', '--model', type=str, default='bert-base-uncased', help='Name of the transformer [bert-base-uncased | '
                                                                     'roberta-base]')
    parser.add_argument(
        '-x', '--xtype', type=str, default='bert', help='Model type [bert | roberta]')
    parser.add_argument(
        '-c', '--cwa', type=str, default="all", help='[cwa | not_cwa | all]')
    args = parser.parse_args()

    net_backward = BackwardChainingInference(teacher_model=args.model, model_type=args.xtype, unification_model=None).to(device)
    net_ruletaker = Transformer(model=args.model, model_type=args.xtype).to(device)

    all_select = ast.literal_eval(args.select)

    tuned_w = {}

    for s in all_select:
        weight = tune(args.batch, args.root, ast.literal_eval(args.depths), [s],
                        args.teacher, args.unification, args.ruletaker, args.name_data, args.cwa)
        tuned_w[f"{s}"] = weight

    filename = f"tuned_weights_ensb_unifier_{os.path.basename(args.unification)}_teacher_" \
               f"{os.path.basename(args.teacher)}_rt_{os.path.basename(args.ruletaker)}_data_{args.name_data}_{args.cwa}.dict"
    with open(os.path.join("./", filename), "w+") as f:
        json.dump(tuned_w, f, indent=4)

