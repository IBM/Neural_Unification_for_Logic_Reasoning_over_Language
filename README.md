# Neural Unification for Logic Reasoning over Language

Implementation of the method presented in the paper: [Neural Unification for Logic Reasoning over Language](https://arxiv.org/abs/2109.08460)


## Introduction 

Automated Theorem Proving (ATP) deals with the development of computer programs being able to show that some conjectures (queries) are a logical consequence of a set of axioms (facts and rules).
There exists several successful ATPs where conjectures and axioms are formally provided (e.g. formalised as First Order Logic formulas).
Recent approaches, such as  (Clark et al., 2020), have proposed transformer-based architectures for deriving conjectures given axioms expressed in natural language (English).
The conjecture is verified through a binary text classifier, where the transformers model is trained to predict the truth value of a conjecture given the axioms.
The RuleTaker approach of  (Clark et al., 2020) achieves appealing results both in terms of accuracy and in the ability to generalize, showing that when the
model is trained with deep enough queries (at least 3 inference steps), the transformers are able to correctly answer the majority of queries (97.6%) that require up to 5 inference steps.
In this work we propose a new architecture, namely the **Neural Unifier**, and a relative training procedure, which achieves state-of-the-art results in term of generalisation, 
showing that mimicking a well-known inference procedure, the **backward chaining**, it is possible to answer deep queries even when the model is trained only on
shallow ones. 
The approach is demonstrated in experiments using a diverse set of benchmark data and the source code is released to the research community for reproducibility.

![alt text](https://i.imgur.com/VirJdqa.png)

In this paper we propose an architecture that is able to answer  deep queries (having large inference depths) even when it is trained only with shallow queries at depth 1 or 2. 
Our main assumption is that by inducing a neural network to mimic some elements of an explicit general reasoning procedure, e.g. the **backward chaining**,  we can increase the ability of the model to generalize. In particular we focus on mimicking the iterative process in which, at each step, a query is simplified by unifying it with an existing rule to create a new but simpler query for further checking. 
In a unification step, when the query matches with the consequent (*Then* clause) of a rule, the antecedent (*If* clause) of the rule is combined with that query via symbolic substitution to create a new query. For example, for the query ``Bob is green" shown in the following figure, the following steps lead to the answer (proof):

- Fact checking step 0: No fact in our knowledge base matches with the query ``Bob is green"
- Unification step 1: From the query: ``Bob is green" and the  rule: ``If someone is smart then it is also green.", a new query is created ``Bob is smart"
- Fact checking step 2: No fact matches with the new query ``Bob is smart"  
- Unification step 3: From the query: ``Bob is green" and the  rule: ``If someone is rough then it is also green", a new query ``Bob is rough" is created
- Fact checking step 4: ``Bob is rough" matches with a fact in the knowledge base, the proof completes and returns the answer: ``Bob is green" is a true statement.

As we can see in the given example, the query ``Bob is green" is simplified iteratively with the help of the unification steps and is transformed into a factual query ``Bob is rough", which is then checked by the fact-checking step via a simple look-up in the knowledge base. These sequences of inference steps are the basis of the famous backward chaining inference in formal logic illustrated in the following figure: 

![alt text](https://i.imgur.com/JWP8tZ2.png)

### Running Example

Train the fact look-up unit on queries with depth = 0:    

```python -m spectre.train_fact_lookup_unit -r ./rule-reasoning-dataset-V2020.2.4 -b 8 -l 0.00001 -d '[0]' -s '[0]'```

Test the fact look-up unit performance on unseen queries at depth 0:

```python -m spectre.test_fact_lookup_unit -r ./rule-reasoning-dataset-V2020.2.4 -t './fact_lookup_[0]_[0].model' -b 8 -d '[5]' -s '[0]'```

The accuracy should be around `1`

Train the Neural Unification model to unify queries at depth 2 using the previously trained fact look-up unit:

```python -m spectre.train_backward_chaining_inference -r ./rule-reasoning-dataset-V2020.2.4 -b 8 -l 0.00001 -t './fact_lookup_[0]_[0].model' -d '[2]' -s '[2]'```

Test the Neural Unifier performance on unseen queries at depth 5:

```python -m spectre.test_backward_chaining_inference -r ./rule-reasoning-dataset-V2020.2.4 -b 8 -t './fact_lookup_[0]_[0].model' -u './unification_[2]_[2].model' -d '[5]' -s '[5]'```

The accuracy should be around `0.95`, as reported in the paper in table 1 (column with model NU (d = 2)).

By changing the depth of the queries (with the -s parameter) the model achieve results similar to the following table (figure 4 of the paper):

| Depth | Accuracy % |
| --- | --- |
| 0 | 34.4 |
| 1 | 44 |
| 2 | 64.9 |
| 3 | 79.2 |
| 4 | 88 |
| 5 | 95 |

Test the Neural Unifier performance on unseen provable queries at depth 0:

```python -m spectre.test_backward_chaining_inference -r ./rule-reasoning-dataset-V2020.2.4 -b 8 -t './fact_lookup_[0]_[0].model' -u './unification_[2]_[2].model' -d '[5]' -s '[5]' -c not_cwa```

The accuracy should be around `0.946`, as reported in the paper in table 3 (column with model NU (d = 2)).


Test the Neural Unifier performance on unseen paraphrased provable queries at depth 0:

```python -m spectre.test_backward_chaining_inference -r ./rule-reasoning-dataset-V2020.2.4/NatLang -b 8 -t './fact_lookup_[0]_[0].model' -u './unification_[2]_[2].model' -d '[5]' -s '[5]' -c not_cwa -n nature```

The accuracy should be around `1`, as reported in the paper in table 4 (column with model NU (d = 2)).

### Training 

#### Train fact lookup unit

```
python train_fact_lookup_unit.py 
  -r ROOT*, --root ROOT  Root directory with the training data
  -d DEPTHS, --depths DEPTHS
                        Reasoning depths
  -l LR, --lr LR        Learning rate
  -e EPOCHS, --epochs EPOCHS
                        The number of epochs
  -b BATCH, --batch BATCH
                        Mini-batch size
  -s SELECT, --select SELECT
                        Whether only consider only questions with the given
                        selected depths
  -o OUTPUT, --output OUTPUT
                        Output folder where the model will be pickled
  -m MODEL, --model MODEL
                        Name of the transformer [bert-base-uncased | roberta-
                        base]
  -x XTYPE, --xtype XTYPE
                        Model type [bert | roberta]
  -y YPRETRAIN, --ypretrain YPRETRAIN
                        Pretrained base model

  *: required
```

#### Train Neural Unification model

```
python train_backward_chaining_inference.py 
  -r ROOT*, --root ROOT  Root directory with the training data
  -d DEPTHS, --depths DEPTHS
                        Reasoning depths
  -l LR, --lr LR        Learning rate
  -e EPOCHS, --epochs EPOCHS
                        The number of epochs
  -b BATCH, --batch BATCH
                        Mini-batch size
  -s SELECT, --select SELECT
                        Whether only consider only questions with the given
                        selected depths
  -t TEACHER, --teacher TEACHER
                        Location of the teacher model
  -o OUTPUT, --output OUTPUT
                        Output folder where the model will be pickled
  -m MODEL, --model MODEL
                        Name of the transformer [bert-base-uncased | roberta-
                        base]  
  -x XTYPE, --xtype XTYPE
                        Model type [bert | roberta]
  -y YPRETRAIN, --ypretrain YPRETRAIN
                        Pretrain model
  *: required
```

### Testing

#### Test the fact lookup unit

```
python test_fact_lookup_unit.py 
  -r ROOT*, --root ROOT  Root directory with the training data
  -d DEPTHS, --depths DEPTHS
                        Reasoning depths
  -b BATCH, --batch BATCH
                        Mini-batch size
  -s SELECT, --select SELECT
                        Whether only consider only questions with the given
                        selected depths
  -t TEACHER, --teacher TEACHER
                        Location of the teacher model
  -o OUTPUT, --output OUTPUT
                        Output folder where the model will be pickled
  -m MODEL, --model MODEL
                        Name of the transformer [bert-base-uncased | roberta-
                        base]  
  -x XTYPE, --xtype XTYPE
                        Model type [bert | roberta]
  -y YPRETRAIN, --ypretrain YPRETRAIN
                        Pretrain model
  -c CWA, --cwa CWA
                        Select the type of queries: [cwa | not_cwa | all]')
  -n NAME_DATA, --name_data NAME_DATA
                        Data set name: synthetic, nature, electricity
  *: required
```

#### Test the Neural Unification model

```
python test_backward_chaining_inference.py 
  -r ROOT*, --root ROOT  Root directory with the training data
  -d DEPTHS, --depths DEPTHS
                        Reasoning depths
  -l LR, --lr LR        Learning rate
  -e EPOCHS, --epochs EPOCHS
                        The number of epochs
  -b BATCH, --batch BATCH
                        Mini-batch size
  -s SELECT, --select SELECT
                        Whether only consider only questions with the given
                        selected depths
  -t TEACHER, --teacher TEACHER
                        Location of the teacher model
  -u UNIFICATION, --unification UNIFICATION
                        Location of the unification model
  -n NAME_DATA, --name_data NAME_DATA
                        Data set name: synthetic, nature,
                        AttPosElectricityRB[1-4], AttPosBirdsVar[1-2]-3
  -m MODEL, --model MODEL
                        Name of the transformer [bert-base-uncased | roberta-
                        base]
  -x XTYPE, --xtype XTYPE
                        Model type [bert | roberta]
  -c CWA, --cwa CWA
                        Select the type of queries: [cwa | not_cwa | all]')
  -n NAME_DATA, --name_data NAME_DATA
                        Data set name: synthetic, nature, electricity

  *: required
```
* -m MODEL and -x XTYPE must be the same as those used during the model training 
