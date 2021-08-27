import argparse
import ast
import os
from pathlib import Path

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import RuleReasoningDataset, RuleReasoningDatasetOnlyCWA, RuleReasoningDatasetOnlyNotCWA, CWA
from .module.transformers import Transformer
from .module.utils.utils import grad_stats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(epoch, batch_size, root, depth, select, cwa):
    if cwa == CWA.ALL:
        rules_data_set = RuleReasoningDataset(root, depth, 'train', select)
    elif cwa == CWA.CWA:
        rules_data_set = RuleReasoningDatasetOnlyCWA(root, depth, 'train', select)
    elif cwa == CWA.NOT_CWA:
        rules_data_set = RuleReasoningDatasetOnlyNotCWA(root, depth, 'train', select)
    train_set = DataLoader(
        rules_data_set, batch_size=batch_size, num_workers=4, shuffle=True
    )

    dataset = iter(train_set)
    pbar = tqdm(dataset)
    moving_loss = 0
    net.train()
    for context, question, answer in pbar:
        answer = answer.to(device)
        net.zero_grad()
        embeds, masks, types = net.data_prep(context, question)
        loss, output = net(embeds, masks, types, answer)
        loss.backward()
        optimizer.step()
        correct = (output.detach().argmax(1) == answer.detach()).sum() / (batch_size + 0.0)
        g = grad_stats(net.model)
        # stats(net.classifier)
        if moving_loss == 0:
            moving_loss = correct

        else:
            moving_loss = moving_loss * 0.99 + correct * 0.01

        pbar.set_description(
            'Epoch: {}; Loss: {:.5f}; Acc: {:.5f}; Grad: {:.10f}'.format(
                epoch + 1, loss.item(), moving_loss, g
            )
        )


def valid(epoch, batch_size, root, depth, select):
    rules_data_set = RuleReasoningDataset(root, depth, 'dev', select)
    valid_set = DataLoader(
        rules_data_set, batch_size=batch_size, num_workers=4
    )
    dataset = iter(valid_set)
    correct = 0.0
    n = 0.0
    net.eval()
    with torch.no_grad():
        for context, question,  answer in tqdm(dataset):
            embeds, masks, types = net.data_prep(context, question)
            loss, output = net(embeds, masks, types, answer)
            correct += (output.detach().argmax(1) == answer.to(device)).sum()
            n += len(output)

    accuracy = correct/n
    print(
        'Epoch: {}; Avg Acc: {:.5f}'.format(
           epoch + 1, correct/n
        )
    )
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer for reasoning')
    parser.add_argument(
        '-r', '--root', required=True, type=str, help='Root directory with the training data')
    parser.add_argument(
        '-d', '--depths', type=str, default='[3]', help='Reasoning depths')
    parser.add_argument(
        '-l', '--lr',  type=float, default=10e-3, help='Learning rate')
    parser.add_argument(
        '-e', '--epochs', type=int, default=3, help='The number of epochs')
    parser.add_argument(
        '-b', '--batch', type=int, default=64, help='Mini-batch size')
    parser.add_argument(
        '-s', '--select', type=str, default='[]',
        help='Whether only consider only questions with the given selected depths')
    parser.add_argument(
        '-o', '--output', type=str, default='./', help='Output folder where the model will be pickled')
    parser.add_argument(
        '-m', '--model', type=str, default='bert-base-uncased', help='Name of the transformer [bert-base-uncased | '
                                                                     'roberta-base]')
    parser.add_argument(
        '-x', '--xtype', type=str, default='bert', help='Model type [bert | roberta]')
    parser.add_argument(
        '-y', '--ypretrain', type=str, default=None, help='Pretrained base model path')

    parser.add_argument(
        '-c', '--cwa', type=str, default="all", help='[cwa | not_cwa | all]')

    args = parser.parse_args()

    net = Transformer(model=args.model, model_type=args.xtype, pretrained_model=args.ypretrain).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    best_so_far = 0.0
    early_stop = 0
    for epoch in range(args.epochs):
        train(epoch, args.batch, args.root, ast.literal_eval(args.depths), ast.literal_eval(args.select), args.cwa)
        accuracy = valid(epoch, args.batch, args.root, ast.literal_eval(args.depths), ast.literal_eval(args.select))
        if best_so_far < accuracy:
            print("The validation accuracy has improved, pickle the best model!", best_so_far,  accuracy.item())
            best_so_far = accuracy
            checkpoint_dir = args.output
            Path(checkpoint_dir).mkdir(exist_ok=True)
            with open(os.path.join(checkpoint_dir, f'fact_lookup_{args.depths}_{args.select}.model'), 'wb+') as f:
                torch.save(net.state_dict(), f)
            early_stop = 0
        else:
            early_stop += 1

        if early_stop >= 2:
            break
