import enum
import random

from torch.utils.data import Dataset
import json


class CWA(str, enum.Enum):
    ALL = "all",
    CWA = "cwa",
    NOT_CWA = "not_cwa",


class RuleReasoningDataset(Dataset):
    def __init__(self, root, depths, split='train', select_depth=None):
        self.data = []
        self.label_dict = {True: 1, False: 0}
        for depth in depths:
            with open(f'{root}/depth-{depth}/{split}.jsonl', 'r') as f:
                lines = [json.loads(jline) for jline in f.read().split('\n')]
            for line in enumerate(lines):
                context = line[1]['context']
                questions = line[1]['questions']
                for question in questions:
                    x = question['meta']['QDep']
                    d = (context, question['text'],  question['label'])
                    if select_depth is not None:
                        if x in select_depth:
                            self.data.append(d)
                    else:
                        self.data.append(d)

        print("Data size:", len(self.data))

    def __getitem__(self, index):
        context, question, answer = self.data[index]
        return context, question,  self.label_dict[answer]

    def __len__(self):
        return len(self.data)


class RuleReasoningDatasetForCWAClassification(Dataset):
    def __init__(self, root, depths, split='train', select_depth=None):
        self.data = []
        self.label_dict = {True: 1, False: 0}
        for depth in depths:
            with open(f'{root}/depth-{depth}/{split}.jsonl', 'r') as f:
                lines = [json.loads(jline) for jline in f.read().split('\n')]
            for line in enumerate(lines):
                context = line[1]['context']
                questions = line[1]['questions']
                for question in questions:
                    x = question['meta']['QDep']
                    label = not ("proof" in question['meta']['strategy'] or "inv-proof" in question['meta']['strategy'])
                    d = (context, question['text'],  label)
                    if select_depth is not None:
                        if x in select_depth:
                            self.data.append(d)
                    else:
                        self.data.append(d)

        print("Data size:", len(self.data))

    def __getitem__(self, index):
        context, question, answer = self.data[index]
        return context, question,  self.label_dict[answer]

    def __len__(self):
        return len(self.data)


class RuleReasoningDatasetOnlyCWA(Dataset):
    def __init__(self, root, depths, split='train', select_depth=None):
        self.data = []
        self.label_dict = {True: 1, False: 0}
        for depth in depths:
            with open(f'{root}/depth-{depth}/{split}.jsonl', 'r') as f:
                lines = [json.loads(jline) for jline in f.read().split('\n')]
            for line in enumerate(lines):
                context = line[1]['context']
                questions = line[1]['questions']
                for question in questions:
                    x = question['meta']['QDep']
                    if not ("proof" in question['meta']['strategy'] or "inv-proof" in question['meta']['strategy']):
                        d = (context, question['text'],  question['label'])
                        if select_depth is not None:
                            if x in select_depth:
                                self.data.append(d)
                        else:
                            self.data.append(d)

        print("Data size:", len(self.data))

    def __getitem__(self, index):
        context, question, answer = self.data[index]
        return context, question,  self.label_dict[answer]

    def __len__(self):
        return len(self.data)


class RuleReasoningDatasetOnlyNotCWA(Dataset):
    def __init__(self, root, depths, split='train', select_depth=None):
        self.data = []
        self.label_dict = {True: 1, False: 0}
        for depth in depths:
            with open(f'{root}/depth-{depth}/{split}.jsonl', 'r') as f:
                lines = [json.loads(jline) for jline in f.read().split('\n')]
            for line in enumerate(lines):
                context = line[1]['context']
                questions = line[1]['questions']
                for question in questions:
                    x = question['meta']['QDep']
                    if "proof" in question['meta']['strategy'] or "inv-proof" in question['meta']['strategy']:
                        d = (context, question['text'],  question['label'])
                        if select_depth is not None:
                            if x in select_depth:
                                self.data.append(d)
                        else:
                            self.data.append(d)

        print("Data size:", len(self.data))

    def __getitem__(self, index):
        context, question, answer = self.data[index]
        return context, question,  self.label_dict[answer]

    def __len__(self):
        return len(self.data)


class RuleReasoningDatasetAllBalanced(Dataset):
    def __init__(self, root, depths, split='train', select_depth=None):
        self.data = []
        self.label_dict = {True: 1, False: 0}
        for depth in depths:
            with open(f'{root}/depth-{depth}/{split}.jsonl', 'r') as f:
                lines = [json.loads(jline) for jline in f.read().split('\n')]
            for line in enumerate(lines):
                context = line[1]['context']
                questions = line[1]['questions']
                for question in questions:
                    x = question['meta']['QDep']
                    if "proof" in question['meta']['strategy'] or "inv-proof" in question['meta']['strategy']:
                        d = (context, question['text'], question['label'])
                        if select_depth is not None:
                            if x in select_depth:
                                self.data.append(d)
                        else:
                            self.data.append(d)
        not_false_queries = list(filter(lambda x: "not" in x[1] and x[2] is False, self.data))
        not_true_queries = list(filter(lambda x: "not" in x[1] and x[2] is True, self.data))
        self.data = [item for item in self.data if item not in not_false_queries and item not in not_true_queries]
        self.__balance_negative_queries_lists(not_false_queries, not_true_queries)
        not_not_true_queries = list(filter(lambda x: "not" not in x[1] and x[2] is True, self.data))
        n_not_not_false_queries = len(list(filter(lambda x: "not" not in x[1] and x[2] is False, self.data)))
        self.data = [item for item in self.data if item not in not_not_true_queries]
        random.shuffle(not_not_true_queries)
        self.data.extend(not_not_true_queries[0:n_not_not_false_queries])
        print("Data size:", len(self.data))

    def __balance_negative_queries_lists(self, queries_1, queries_2):
        if len(queries_2) > len(queries_1):
            queries_1, queries_2 = queries_2, queries_1
        random.shuffle(queries_1)
        while len(queries_1) > len(queries_2):
            e = queries_1.pop()
            new_query = e[1].replace("not", "")
            new_label = not e[2]
            self.data.append((e[0], new_query, new_label))
        self.data.extend(queries_1)
        self.data.extend(queries_2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        context, question, answer = self.data[index]
        return context, question,  self.label_dict[answer]

# class RuleReasoningDatasetAllBalanced(Dataset):
#     def __init__(self, root, depths, split='train', select_depth=None):
#         self.data = []
#         self.label_dict = {True: 1, False: 0}
#         for depth in depths:
#             with open(f'{root}/depth-{depth}/{split}.jsonl', 'r') as f:
#                 lines = [json.loads(jline) for jline in f.read().split('\n')]
#             for line in enumerate(lines):
#                 context = line[1]['context']
#                 questions = line[1]['questions']
#                 for question in questions:
#                     x = question['meta']['QDep']
#                     if "proof" in question['meta']['strategy'] or "inv-proof" in question['meta']['strategy']:
#                         d = (context, question['text'], question['label'])
#                         if select_depth is not None:
#                             if x in select_depth:
#                                 self.data.append(d)
#                         else:
#                             self.data.append(d)
#         with_not_false_queries = list(filter(lambda x: "not" in x[1] and x[2] is False, self.data))
#         with_not_true_queries = list(filter(lambda x: "not" in x[1] and x[2] is True, self.data))
#         without_not_true_queries = list(filter(lambda x: "not" not in x[1] and x[2] is True, self.data))
#         without_not_false_queries = list(filter(lambda x: "not" not in x[1] and x[2] is False, self.data))
#         self.data = []
#         random.shuffle(without_not_true_queries)
#         ratio = 0.1
#         n = int(ratio*len(without_not_false_queries))
#         self.data.extend(without_not_true_queries[0:n])
#         self.data.extend(without_not_false_queries)
#
#         random.shuffle(with_not_false_queries)
#         n = int(ratio * len(with_not_true_queries))
#         self.data.extend(with_not_false_queries[0:n])
#         self.data.extend(with_not_true_queries)
#
#         print("Data size:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        context, question, answer = self.data[index]
        return context, question,  self.label_dict[answer]


class RuleReasoningDatasetNotCWATrueBalanced(Dataset):
    def __init__(self, root, depths, split='train', select_depth=None):
        self.data = []
        self.label_dict = {True: 1, False: 0}
        for depth in depths:
            with open(f'{root}/depth-{depth}/{split}.jsonl', 'r') as f:
                lines = [json.loads(jline) for jline in f.read().split('\n')]
            for line in enumerate(lines):
                context = line[1]['context']
                questions = line[1]['questions']
                for question in questions:
                    x = question['meta']['QDep']
                    if "proof" in question['meta']['strategy'] or "inv-proof" in question['meta']['strategy']:
                        d = (context, question['text'], question['label'])
                        if select_depth is not None:
                            if x in select_depth:
                                self.data.append(d)
                        else:
                            self.data.append(d)
        not_false_queries = list(filter(lambda x: "not" in x[1] and x[2] is False, self.data))
        not_true_queries = list(filter(lambda x: "not" in x[1] and x[2] is True, self.data))
        self.data = [item for item in self.data if item not in not_false_queries and item not in not_true_queries]
        self.__balance_negative_queries_lists(not_false_queries, not_true_queries)
        print("Data size:", len(self.data))

    def __balance_negative_queries_lists(self, queries_1, queries_2):
        if len(queries_2) > len(queries_1):
            queries_1, queries_2 = queries_2, queries_1
        random.shuffle(queries_1)
        while len(queries_1) > len(queries_2):
            e = queries_1.pop()
            new_query = e[1].replace("not", "")
            new_label = not e[2]
            self.data.append((e[0], new_query, new_label))
        self.data.extend(queries_1)
        self.data.extend(queries_2)

    def __getitem__(self, index):
        context, question, answer = self.data[index]
        return context, question,  self.label_dict[answer]

    def __len__(self):
        return len(self.data)


class BirdElectricityDataset(Dataset):
    def __init__(self, root, split='train', select_depth=None, name_data=None, cwa=CWA.ALL):
        self.data = []
        self.label_dict = {True: 1, False: 0}

        with open(f'{root}/{split}.jsonl', 'r') as f:
            lines = [json.loads(jline) for jline in f.read().split('\n')]
        for line in enumerate(lines):
            context = line[1]['context']
            questions = line[1]['questions']
            for question in questions:
                x = question['meta']['QDep']
                d = (context, question['text'],  question['label'])
                is_provable = "proof" in question['meta']['strategy'] or "inv-proof" in question['meta']['strategy']
                if cwa == CWA.ALL or (cwa == CWA.NOT_CWA and is_provable) or (cwa == CWA.CWA and not is_provable):
                    if len(select_depth) > 0:
                        if x in select_depth:
                            self.data.append(d)
                    else:
                        self.data.append(d)

        print("Data size:", len(self.data))

    def __getitem__(self, index):
        context, question, answer = self.data[index]
        return context, question,  self.label_dict[answer]

    def __len__(self):
        return len(self.data)


class NatLangDataset(Dataset):
    def __init__(self, root, split='train', select_depth=None, cwa=CWA.ALL):
        self.data = []
        self.label_dict = {True: 1, False: 0}

        with open(f'{root}/{split}.jsonl', 'r') as f:
            lines = [json.loads(jline) for jline in f.read().split('\n')]
        for line in enumerate(lines):
            context = line[1]['context']
            questions = line[1]['questions']
            for question in questions:
                x = question['meta']['QDep']
                d = (context, question['text'],  question['label'])
                is_provable = "proof" in question['meta']['strategy'] or "inv-proof" in question['meta']['strategy']
                if cwa == CWA.ALL or (cwa == CWA.NOT_CWA and is_provable) or (cwa == CWA.CWA and not is_provable):
                    if len(select_depth) > 0:
                        if x in select_depth:
                            self.data.append(d)
                    else:
                        self.data.append(d)

        print("Data size:", len(self.data))

    def __getitem__(self, index):
        context, question, answer = self.data[index]
        return context, question,  self.label_dict[answer]

    def __len__(self):
        return len(self.data)