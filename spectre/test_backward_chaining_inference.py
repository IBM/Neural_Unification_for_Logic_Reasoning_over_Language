import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .datasets import RuleReasoningDataset, BirdElectricityDataset, NatLangDataset, CWA, RuleReasoningDatasetOnlyCWA, \
    RuleReasoningDatasetOnlyNotCWA
from .module.backward_chaining_inference import BackwardChainingInference
from .module.utils.utils import decode_nearest
import argparse
import ast
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(batch_size, root, depth, select, teacher, unification, name_data, cwa):
    if name_data == 'synthetic':
        if cwa == CWA.ALL:
            data_set = RuleReasoningDataset(root, depth, 'test', select)
        elif cwa == CWA.CWA:
            data_set = RuleReasoningDatasetOnlyCWA(root, depth, 'test', select)
        elif cwa == CWA.NOT_CWA:
            data_set = RuleReasoningDatasetOnlyNotCWA(root, depth, 'test', select)
    elif name_data == 'nature':
        data_set = NatLangDataset(root, 'test', select, cwa=cwa)
    else:
        data_set = BirdElectricityDataset(root, 'test', select, name_data, cwa=cwa)
    valid_set = DataLoader(
        data_set, batch_size=batch_size, num_workers=4
    )
    dataset = iter(valid_set)
    correct = 0.0
    n = 0.0

    print('Load unification model from', unification)
    net.load_state_dict(torch.load(unification))

    print('Load teacher model from', teacher)
    net.teacher.load_state_dict(torch.load(teacher))
    net.eval()

    with torch.no_grad():
        i = 0
        for context, question,  answer in tqdm(dataset):
            loss, output, embeds, masks, types = net(context, question, answer)
            correct += (output.detach().argmax(1) == answer.to(device)).sum()
            n += len(output)
            if i < 10:
                print(decode_nearest(embeds[0], net.teacher.base_model.embeddings.word_embeddings.weight,
                                     net.teacher.tokenizer, masks[0], types[0]), '[QUES]', question[0], answer[0].item(), output.detach().argmax(1)[0].item())
            i += 1

    print(
        'Average Accuracy: {:.5f}'.format(
            correct/n
        )
    )
    return correct/n


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer for reasoning')
    parser.add_argument(
        '-r', '--root', required=True, type=str, help='Root directory with the training data')
    parser.add_argument(
        '-t', '--teacher', type=str, required=True, help='Location of the teacher model')
    parser.add_argument(
        '-u', '--unification', type=str, required=True, help='Location of the unification model')
    parser.add_argument(
        '-d', '--depths', type=str, default='[3]', help='Reasoning depths')
    parser.add_argument(
        '-l', '--lr',  type=float, default=10e-3, help='Learning rate')
    parser.add_argument(
        '-e', '--epochs', type=int, default=20, help='The number of epochs')
    parser.add_argument(
        '-b', '--batch', type=int, default=64, help='Mini-batch size')
    parser.add_argument(
        '-s', '--select', type=str, default='[]',
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

    net = BackwardChainingInference(teacher_model=args.model, model_type=args.xtype, unification_model=None).to(device)
    accuracy = test(args.batch, args.root, ast.literal_eval(args.depths), ast.literal_eval(args.select),
                    args.teacher, args.unification, args.name_data, args.cwa).item()

    filename = f"log_nu_unifier_{os.path.basename(args.unification)}_teacher_" \
               f"{os.path.basename(args.teacher)}_data_{args.name_data}_{args.cwa}"
    with open(os.path.join("./", filename), "a+") as f:
        f.write(f"\nD {args.depths}  , S {args.select} : {accuracy}")
