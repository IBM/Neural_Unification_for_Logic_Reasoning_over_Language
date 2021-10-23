# Datasets for rule reasoning

This directory contains the various datasets associated with the paper "Transformers as Soft Reasoners over Language" (arXiv Feb 2020).

The dataset directories are as follows:

  * depth-N (N=0, 1, 2, 3, 5): Questions with rulebases in synthetic language reasoning paths up to depth N, as defined in the paper.
  * birds-electricity: Questions with "birds" and "electricity" rulebases.
  * NatLang: Questions with rulebases in crowdsourced natural language.
  * depth-3ext: The Depth3 dataset augmented with 10% each of the depth=0, depth=1, and depth-2 datasets.
  * depth-3ext-NatLang: The Depth3Ext dataset augmented with the NatLang one.
  * alternate-formats: Each of the datasets in an alternate TSV format.

Except for the alternate-formats directory, each dataset is in the format of two JSONL files, named `train.jsonl` and `meta-train.jsonl` 
(for train split, similar for dev and test). 

There is a train/dev/test split for each of the dataset (except birds-electricity), where each unique rulebase is contained in a single split. In addition, for the max depth-5 dataset there is a special "test-xtra" split which contains a new set of rulebases where every possible statement is included.

Except for the extended datasets, the number of questions in each split is around 70k/10k/20k for train/dev/test.


## Format of `[train|dev|test].jsonl` files

Here is a sample entry from a `train.jsonl` file (with whitespace added for clarity):

```
{
 "id":"RelNeg-D2-1742",
 "context":"If the squirrel eats the bald eagle and the bald eagle does not visit the squirrel then the 
bald eagle needs the squirrel. Green things are rough. If something eats the squirrel and it needs the 
bald eagle then it does not need the squirrel. The bald eagle is nice. If something needs the bald eagle 
and it is not cold then it visits the bald eagle. If the bald eagle eats the squirrel and something does 
not eat the bald eagle then the bald eagle visits the squirrel. If the squirrel is rough and the bald eagle 
does not need the squirrel then the squirrel needs the bald eagle. If something is nice and it does not eat 
the squirrel then the squirrel eats the bald eagle. If something eats the squirrel then the squirrel needs 
the bald eagle. The squirrel visits the bald eagle.",
 "meta":{"sentenceScramble":[9,5,6,1,3,8,10,4,7,2]},
 "questions":[
   {"id":"RelNeg-D2-1742-1","text":"The squirrel visits the bald eagle.","label":true,"meta":{"QDep":0,"QLen":1,"strategy":"proof","Qid":"Q1"}},
   {"id":"RelNeg-D2-1742-2","text":"The bald eagle is not nice.","label":false,"meta":{"QDep":0,"QLen":1,"strategy":"inv-proof","Qid":"Q2"}},
   {"id":"RelNeg-D2-1742-3","text":"The squirrel eats the bald eagle.","label":true,"meta":{"QDep":1,"QLen":3,"strategy":"proof","Qid":"Q3"}},
   {"id":"RelNeg-D2-1742-4","text":"The squirrel does not eat the bald eagle.","label":false,"meta":{"QDep":1,"QLen":3,"strategy":"inv-proof","Qid":"Q4"}},
   {"id":"RelNeg-D2-1742-5","text":"The bald eagle needs the squirrel.","label":true,"meta":{"QDep":2,"QLen":5,"strategy":"proof","Qid":"Q5"}},
   {"id":"RelNeg-D2-1742-6","text":"The bald eagle does not need the squirrel.","label":false,"meta":{"QDep":2,"QLen":5,"strategy":"inv-proof","Qid":"Q6"}},
   {"id":"RelNeg-D2-1742-7","text":"The bald eagle does not visit the bald eagle.","label":true,"meta":{"QDep":1,"QLen":"","strategy":"inv-rconc","Qid":"Q7"}},
   {"id":"RelNeg-D2-1742-8","text":"The bald eagle visits the squirrel.","label":false,"meta":{"QDep":1,"QLen":"","strategy":"rconc","Qid":"Q8"}},
   {"id":"RelNeg-D2-1742-9","text":"The squirrel is not rough.","label":true,"meta":{"QDep":1,"QLen":"","strategy":"inv-rconc","Qid":"Q9"}},
   {"id":"RelNeg-D2-1742-10","text":"The squirrel visits the squirrel.","label":false,"meta":{"QDep":0,"QLen":"","strategy":"random","Qid":"Q10"}},
   {"id":"RelNeg-D2-1742-11","text":"The squirrel is not green.","label":true,"meta":{"QDep":0,"QLen":"","strategy":"inv-random","Qid":"Q11"}},
   {"id":"RelNeg-D2-1742-12","text":"The squirrel needs the squirrel.","label":false,"meta":{"QDep":0,"QLen":"","strategy":"random","Qid":"Q12"}}
  ]
}
```

Description of the `train.jsonl` fields: 

* `id`: The rulebase ID, where in the example above "RelNeg" means it's a rulebase with relations and negations, "D2" means it's from the depth-2 dataset, 1742 is the index to the rulebase (see below).
* `context`: The rulebase expressed in (synthetic) language.
* `meta`-`sentenceScramble`: How the original list of triples and rules were reordered to form the rulebase context.
* `questions`: The list of questions associated with the rulebase, with fields:
  * `id`: Individual question id.
  * `text`: The (synthetic) language of the question.
  * `label`: Whether the question statement is `true` or `false` according to the rulebase.
  * `meta`: Metadata for the question: 
      * `QDep`: The "depth" of the "proof" for the question (see paper for details).
      * `QLen`: The "length" of the "proof" (number of leaves).
      * `Qid`: The index of the question as referenced in the metadata file (see below).
      * `strategy`: How the statement was derived, one of:
          * "proof" / "inv-proof": A proven statement (or its negation).
          * "rconc" / "inv-rconc": An unproven rule conclusion in the rulebase (or its negation).
          * "random" / "inv-random": A randomly selected unproven statement not covered by the above (or its negation).

## Format of `meta-[train|dev|test].jsonl` files

Along with the `train.jsonl` file there is additional metadata in a `meta-train.jsonl` file. Here is a sample entry corresponding to the example above:

```{
 "id":"RelNeg-D2-1742", "maxD":2, "NFact":2, "NRule":8,
 "triples":{
  "triple1":{"text":"The bald eagle is nice.","representation":"(\"bald eagle\" \"is\" \"nice\" \"+\")"},
  "triple2":{"text":"The squirrel visits the bald eagle.","representation":"(\"squirrel\" \"visits\" \"bald eagle\" \"+\")"}},
 "rules":{
  "rule1":{"text":"If something needs the bald eagle and it is not cold then it visits the bald eagle.","representation":"(((\"something\" \"needs\" \"bald eagle\" \"+\") (\"something\" \"is\" \"cold\" \"~\")) -> (\"something\" \"visits\" \"bald eagle\" \"+\"))"},
  "rule2":{"text":"If something is nice and it does not eat the squirrel then the squirrel eats the bald eagle.","representation":"(((\"something\" \"is\" \"nice\" \"+\") (\"something\" \"eats\" \"squirrel\" \"~\")) -> (\"squirrel\" \"eats\" \"bald eagle\" \"+\"))"},
  "rule3":{"text":"Green things are rough.","representation":"(((\"something\" \"is\" \"green\" \"+\")) -> (\"something\" \"is\" \"rough\" \"+\"))"},
  "rule4":{"text":"If something eats the squirrel and it needs the bald eagle then it does not need the squirrel.","representation":"(((\"something\" \"eats\" \"squirrel\" \"+\") (\"something\" \"needs\" \"bald eagle\" \"+\")) -> (\"something\" \"needs\" \"squirrel\" \"-\"))"},
  "rule5":{"text":"If something eats the squirrel then the squirrel needs the bald eagle.","representation":"(((\"something\" \"eats\" \"squirrel\" \"+\")) -> (\"squirrel\" \"needs\" \"bald eagle\" \"+\"))"},
  "rule6":{"text":"If the bald eagle eats the squirrel and something does not eat the bald eagle then the bald eagle visits the squirrel.","representation":"(((\"bald eagle\" \"eats\" \"squirrel\" \"+\") (\"something\" \"eats\" \"bald eagle\" \"~\")) -> (\"bald eagle\" \"visits\" \"squirrel\" \"+\"))"},
  "rule7":{"text":"If the squirrel eats the bald eagle and the bald eagle does not visit the squirrel then the bald eagle needs the squirrel.","representation":"(((\"squirrel\" \"eats\" \"bald eagle\" \"+\") (\"bald eagle\" \"visits\" \"squirrel\" \"~\")) -> (\"bald eagle\" \"needs\" \"squirrel\" \"+\"))"},
  "rule8":{"text":"If the squirrel is rough and the bald eagle does not need the squirrel then the squirrel needs the bald eagle.","representation":"(((\"squirrel\" \"is\" \"rough\" \"+\") (\"bald eagle\" \"needs\" \"squirrel\" \"~\")) -> (\"squirrel\" \"needs\" \"bald eagle\" \"+\"))"}},
 "questions":{
  "Q1":{"question":"The squirrel visits the bald eagle.","answer":true,"QDep":0,"QLen":1,"strategy":"proof","proofs":"[(triple2)]","representation":"(\"squirrel\" \"visits\" \"bald eagle\" \"+\")"},
  "Q2":{"question":"The bald eagle is not nice.","answer":false,"QDep":0,"QLen":1,"strategy":"inv-proof","proofs":"[(triple1)]","representation":"(\"bald eagle\" \"is\" \"nice\" \"-\")"},
  "Q3":{"question":"The squirrel eats the bald eagle.","answer":true,"QDep":1,"QLen":3,"strategy":"proof","proofs":"[(((triple1 NAF) -> rule2))]","representation":"(\"squirrel\" \"eats\" \"bald eagle\" \"+\")"},
  "Q4":{"question":"The squirrel does not eat the bald eagle.","answer":false,"QDep":1,"QLen":3,"strategy":"inv-proof","proofs":"[(((triple1 NAF) -> rule2))]","representation":"(\"squirrel\" \"eats\" \"bald eagle\" \"-\")"},
  "Q5":{"question":"The bald eagle needs the squirrel.","answer":true,"QDep":2,"QLen":5,"strategy":"proof","proofs":"[(((((triple1 NAF) -> rule2) NAF) -> rule7))]","representation":"(\"bald eagle\" \"needs\" \"squirrel\" \"+\")"},
  "Q6":{"question":"The bald eagle does not need the squirrel.","answer":false,"QDep":2,"QLen":5,"strategy":"inv-proof","proofs":"[(((((triple1 NAF) -> rule2) NAF) -> rule7))]","representation":"(\"bald eagle\" \"needs\" \"squirrel\" \"-\")"},
  "Q7":{"question":"The bald eagle does not visit the bald eagle.","answer":true,"QDep":1,"QLen":"","strategy":"inv-rconc","proofs":"[@1: The bald eagle visits the bald eagle.[CWA. Example of deepest failure = (rule1 <- FAIL)]]","representation":"(\"bald eagle\" \"visits\" \"bald eagle\" \"-\")"},
  "Q8":{"question":"The bald eagle visits the squirrel.","answer":false,"QDep":1,"QLen":"","strategy":"rconc","proofs":"[@1: The bald eagle visits the squirrel.[CWA. Example of deepest failure = (rule6 <- FAIL)]]","representation":"(\"bald eagle\" \"visits\" \"squirrel\" \"+\")"},
  "Q9":{"question":"The squirrel is not rough.","answer":true,"QDep":1,"QLen":"","strategy":"inv-rconc","proofs":"[@1: The squirrel is rough.[CWA. Example of deepest failure = (rule3 <- FAIL)]]","representation":"(\"squirrel\" \"is\" \"rough\" \"-\")"},
  "Q10":{"question":"The squirrel visits the squirrel.","answer":false,"QDep":0,"QLen":"","strategy":"random","proofs":"[@0: The squirrel visits the squirrel.[CWA. Example of deepest failure = (FAIL)]]","representation":"(\"squirrel\" \"visits\" \"squirrel\" \"+\")"},
  "Q11":{"question":"The squirrel is not green.","answer":true,"QDep":0,"QLen":"","strategy":"inv-random","proofs":"[@0: The squirrel is green.[CWA. Example of deepest failure = (FAIL)]]","representation":"(\"squirrel\" \"is\" \"green\" \"-\")"},
  "Q12":{"question":"The squirrel needs the squirrel.","answer":false,"QDep":0,"QLen":"","strategy":"random","proofs":"[@0: The squirrel needs the squirrel.[CWA. Example of deepest failure = (FAIL)]]","representation":"(\"squirrel\" \"needs\" \"squirrel\" \"+\")"}},
 "allProofs":"@0: The squirrel visits the bald eagle.[(triple2)] The bald eagle is nice.[(triple1)] @1: The squirrel eats the bald eagle.[(((triple1 NAF) -> rule2))] @2: The bald eagle needs the squirrel.[(((((triple1 NAF) -> rule2) NAF) -> rule7))]"
}
```

Description of the `meta-train.jsonl` fields:

* `id`: The id of the rulebase, same as in `train.jsonl` above.
* `maxD`: The max depth of the questions for rulebase.
* `NFact`: The number of facts (triples) in the rulebase.
* `NRule`: The number of rules in the rulebase.
* `triples`: A list of the facts (triples), sequentially identified as triple1, triple2, etc. For each fact:
  * `text`: The (synthetic) language associated with the triple.
  * `representation`: The lisp-format representation of the triple in the form "(arg1 rel arg2 polarity)" where rel is set to "is" for attributes, and polarity is "+" or "-".
  * `rules`: The list of rules in the rulebase, identified as rule1, rule2, etc. For each rule:
    * `text`: The (synthetic) language associated with the rule
    * `representation`: The lisp-format representation of the rule, of the form (lhs1 lhs2 ...) -> rhs 
    representing the meaning "if lhs1 and lhs2 and ... then rhs". Each lhs and rhs is in the same format as the 
    triples above, but can contain the generic "something" argument, while in the lhs a negative polarity is represented 
    as "~" for negation as failure (NAF).
* `questions`: Each of the True/False questions for the rulebase, identified as Q1, Q2, etc. For each question:
  * `question`: The (synthetic) language associated with the question
  * `representation`: The lisp-format representation of the question, same as the triple format described above.
  * `answer`: Specifiying the truth value of the question (`true` or `false`).
  * `QDep`, `QLen`, `strategy`: The depth, length and strategy associated with the question, see description above.
  * `proofs`: The possible proofs for the truth-value of the question, or examples of deepest failure when the closed world 
  assumption (CWA) is invoked. If multiple proof paths are possible, they are separated by "OR".
* `allProofs`: For each depth, this gives all the provable statements in the rulebase along with their proofs

## `alternate-formats` directory

In addition, the alternate-formats directory contains TSV files with the same data as described above. E.g. for the sample rulebase shown above it looks like:

```
ID <t> Context <t> MaxD <t> NFact <t> NRule <t> Question <t> Ans <t> QDep <t> QLen <t> Strategy <t> Proof trees <t> Representation
...
RelNeg1742-theory <t> The bald eagle is nice. The squirrel visits the bald eagle. If something needs the bald eagle and it is not cold then it visits the bald eagle. If something is nice and it does not eat the squirrel then the squirrel eats the bald eagle. Green things are rough. If something eats the squirrel and it needs the bald eagle then it does not need the squirrel. If something eats the squirrel then the squirrel needs the bald eagle. If the bald eagle eats the squirrel and something does not eat the bald eagle then the bald eagle visits the squirrel. If the squirrel eats the bald eagle and the bald eagle does not visit the squirrel then the bald eagle needs the squirrel. If the squirrel is rough and the bald eagle does not need the squirrel then the squirrel needs the bald eagle.  <t> 2 <t> 2 <t> 8 <t> train <t>  <t>  <t>  <t>  <t> @0: The squirrel visits the bald eagle.[(triple2)] The bald eagle is nice.[(triple1)] @1: The squirrel eats the bald eagle.[(((triple1 NAF) -> rule2))] @2: The bald eagle needs the squirrel.[(((((triple1 NAF) -> rule2) NAF) -> rule7))] 
RelNeg1742-triple1 <t> The bald eagle is nice. <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t> ("bald eagle" "is" "nice" "+")
RelNeg1742-triple2 <t> The squirrel visits the bald eagle. <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t> ("squirrel" "visits" "bald eagle" "+")
RelNeg1742-rule1 <t> If something needs the bald eagle and it is not cold then it visits the bald eagle. <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t> ((("something" "needs" "bald eagle" "+") ("something" "is" "cold" "~")) -> ("something" "visits" "bald eagle" "+"))
RelNeg1742-rule2 <t> If something is nice and it does not eat the squirrel then the squirrel eats the bald eagle. <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t> ((("something" "is" "nice" "+") ("something" "eats" "squirrel" "~")) -> ("squirrel" "eats" "bald eagle" "+"))
RelNeg1742-rule3 <t> Green things are rough. <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t> ((("something" "is" "green" "+")) -> ("something" "is" "rough" "+"))
RelNeg1742-rule4 <t> If something eats the squirrel and it needs the bald eagle then it does not need the squirrel. <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t> ((("something" "eats" "squirrel" "+") ("something" "needs" "bald eagle" "+")) -> ("something" "needs" "squirrel" "-"))
RelNeg1742-rule5 <t> If something eats the squirrel then the squirrel needs the bald eagle. <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t> ((("something" "eats" "squirrel" "+")) -> ("squirrel" "needs" "bald eagle" "+"))
RelNeg1742-rule6 <t> If the bald eagle eats the squirrel and something does not eat the bald eagle then the bald eagle visits the squirrel. <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t> ((("bald eagle" "eats" "squirrel" "+") ("something" "eats" "bald eagle" "~")) -> ("bald eagle" "visits" "squirrel" "+"))
RelNeg1742-rule7 <t> If the squirrel eats the bald eagle and the bald eagle does not visit the squirrel then the bald eagle needs the squirrel. <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t> ((("squirrel" "eats" "bald eagle" "+") ("bald eagle" "visits" "squirrel" "~")) -> ("bald eagle" "needs" "squirrel" "+"))
RelNeg1742-rule8 <t> If the squirrel is rough and the bald eagle does not need the squirrel then the squirrel needs the bald eagle. <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t>  <t> ((("squirrel" "is" "rough" "+") ("bald eagle" "needs" "squirrel" "~")) -> ("squirrel" "needs" "bald eagle" "+"))

RelNeg1742-Q1 <t>  <t>  <t>  <t>  <t> The squirrel visits the bald eagle. <t> "TRUE" <t> 0 <t> 1 <t> proof <t> [(triple2)] <t> ("squirrel" "visits" "bald eagle" "+")
RelNeg1742-Q2 <t>  <t>  <t>  <t>  <t> The bald eagle is not nice. <t> "FALSE" <t> 0 <t> 1 <t> inv-proof <t> [(triple1)] <t> ("bald eagle" "is" "nice" "-")
RelNeg1742-Q3 <t>  <t>  <t>  <t>  <t> The squirrel eats the bald eagle. <t> "TRUE" <t> 1 <t> 3 <t> proof <t> [(((triple1 NAF) -> rule2))] <t> ("squirrel" "eats" "bald eagle" "+")
RelNeg1742-Q4 <t>  <t>  <t>  <t>  <t> The squirrel does not eat the bald eagle. <t> "FALSE" <t> 1 <t> 3 <t> inv-proof <t> [(((triple1 NAF) -> rule2))] <t> ("squirrel" "eats" "bald eagle" "-")
RelNeg1742-Q5 <t>  <t>  <t>  <t>  <t> The bald eagle needs the squirrel. <t> "TRUE" <t> 2 <t> 5 <t> proof <t> [(((((triple1 NAF) -> rule2) NAF) -> rule7))] <t> ("bald eagle" "needs" "squirrel" "+")
RelNeg1742-Q6 <t>  <t>  <t>  <t>  <t> The bald eagle does not need the squirrel. <t> "FALSE" <t> 2 <t> 5 <t> inv-proof <t> [(((((triple1 NAF) -> rule2) NAF) -> rule7))] <t> ("bald eagle" "needs" "squirrel" "-")
RelNeg1742-Q7 <t>  <t>  <t>  <t>  <t> The bald eagle does not visit the bald eagle. <t> "TRUE" <t> 1 <t>  <t> inv-rconc <t> [@1: The bald eagle visits the bald eagle.[CWA. Example of deepest failure = (rule1 <- FAIL)]] <t> ("bald eagle" "visits" "bald eagle" "-")
RelNeg1742-Q8 <t>  <t>  <t>  <t>  <t> The bald eagle visits the squirrel. <t> "FALSE" <t> 1 <t>  <t> rconc <t> [@1: The bald eagle visits the squirrel.[CWA. Example of deepest failure = (rule6 <- FAIL)]] <t> ("bald eagle" "visits" "squirrel" "+")
RelNeg1742-Q9 <t>  <t>  <t>  <t>  <t> The squirrel is not rough. <t> "TRUE" <t> 1 <t>  <t> inv-rconc <t> [@1: The squirrel is rough.[CWA. Example of deepest failure = (rule3 <- FAIL)]] <t> ("squirrel" "is" "rough" "-")
RelNeg1742-Q10 <t>  <t>  <t>  <t>  <t> The squirrel visits the squirrel. <t> "FALSE" <t> 0 <t>  <t> random <t> [@0: The squirrel visits the squirrel.[CWA. Example of deepest failure = (FAIL)]] <t> ("squirrel" "visits" "squirrel" "+")
RelNeg1742-Q11 <t>  <t>  <t>  <t>  <t> The squirrel is not green. <t> "TRUE" <t> 0 <t>  <t> inv-random <t> [@0: The squirrel is green.[CWA. Example of deepest failure = (FAIL)]] <t> ("squirrel" "is" "green" "-")
RelNeg1742-Q12 <t>  <t>  <t>  <t>  <t> The squirrel needs the squirrel. <t> "FALSE" <t> 0 <t>  <t> random <t> [@0: The squirrel needs the squirrel.[CWA. Example of deepest failure = (FAIL)]] <t> ("squirrel" "needs" "squirrel" "+")
```
