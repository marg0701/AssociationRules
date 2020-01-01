import pandas as pd
import numpy as np
from efficient_apriori import apriori

data = None
rules = None

def init_model():
    global data
    data = pd.read_csv("dataset/movie_dataset.csv", header=None)
    records = []
    for i in range(0, len(data)):
        records.append([str(data.values[i,j]) for j in range(0, len(data.columns))])

    global rules
    itemsets, rules = apriori(records, min_support=0.1,  min_confidence=0.01)


init_model()


def suggestion(movie, maxrecom):
    title1 = []
    title2 = []
    confidence = []
    results = pd.DataFrame(columns=['Title 1', 'Title 2', 'Confidence'])
    rules_rhs = filter(lambda rule: len(rule.lhs) == 1 and len(rule.rhs) == 1 and rule.lhs[0]!='nan' and rule.rhs[0]!='nan', rules)
    for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
        title1.append(rule.lhs[0])
        title2.append(rule.rhs[0])
        confidence.append(rule.confidence)
    results['Title 1'] = title1
    results['Title 2'] = title2
    results['Confidence'] = confidence

    dfd = results[results['Title 1']==movie].drop_duplicates(subset=['Title 2'])
    dfds = dfd.sort_values(by=['Confidence'], ascending=False)
    suggest = dfds[dfds['Title 2']!='nan']['Title 2'][:maxrecom]
    confs = dfds[dfds['Title 2']!='nan']['Confidence'][:maxrecom]

    return suggest.tolist(), confs.tolist()


def retrain(playlist):
    newdata = data.append(pd.DataFrame([playlist]), ignore_index=True)
    print(newdata.tail())
    newdata.to_csv("dataset/movie_dataset.csv", index=False, header=None)

    init_model()

    return True
