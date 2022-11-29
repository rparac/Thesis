"""
Parse a sentence into its constituencies.
"""
import argparse
import math
import subprocess
from typing import List

import numpy as np

import concept_processing.io
from concept_processing.asp.asp_generator import ASPGenerator
from concept_processing.asp.asp_solver import clingo_solve
from concept_processing.asp.clingo_out_parsers import ClingoExParser, ClingoAnsParser
from concept_processing.enums import ProblemType
from concept_processing.metrics import jaccard, precision, recall
from concept_processing.nlp.nlp_parser import NLPParser
from concept_processing.nlp.nlp_utils import truecase, add_punctuation, merge_not, simple_truecase
from concept_processing.nlp.spacy_wrapper import SpacyWrapper

project_root = {
    "labs": "/vol/bitbucket/rp218/luke-for-roko",
    "home": "/home/rp218/luke-for-roko",
}

base_dir_temp = {
    ProblemType.ATOMISATION: '{}/ilasp/atomisation',
    ProblemType.GENERALISATION: '{}/ilasp/generalisation',
}

concepts_file_temp = {
    ProblemType.ATOMISATION: '{}/created_examples/examples.csv',
    ProblemType.GENERALISATION: '{}/created_examples/examples.csv',
}

premises_out_dir_temp = '{}/premises'
background_knowledge_file_temp = '{}/background.ilasp'
solution_file_temp = '{}/solutions/best_sol.lp'
clingo_out_file_temp = '{}/clingo_out.tmp'
final_output_temp = '{}/generated_sents.csv'


def create_sent_files(dir_path, premise_texts, spacy_lang: NLPParser, problem_type):
    asp_generator = ASPGenerator(spacy_lang, problem_type)
    for i, premise_text in enumerate(premise_texts):
        asp_generator.parse(premise_text)
        with open(clingo_sent_file_name(dir_path, i), 'w') as f:
            for program in asp_generator.get_programs():
                f.write('\n'.join(program))


def clingo_sent_file_name(dir_path, i):
    return f"{dir_path}/sent{i}.lp"


def save_generalised_sents(premise_texts, generated_sents, outfile):
    with open(outfile, 'w') as o:
        for premise_text, sents in zip(premise_texts, generated_sents):
            joined_sents = ' '.join(sents)
            o.write(f'"{premise_text}", "{joined_sents}"\n')


def generate_sentences(premise_texts: List[str], problem_type: ProblemType, premises_out_dir: str,
                       clingo_args: List[str]) -> List[List[str]]:
    ret = []
    for i, premise_text in enumerate(premise_texts):
        clingo_out = clingo_solve(*clingo_args, f"{premises_out_dir}/sent{i}.lp")
        cp = ClingoAnsParser(problem_type)
        sents = cp.get_sentences(clingo_out)
        ret.append(sents)
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate guesses and evaluate solutions')
    parser.add_argument('--eval', type=str, nargs='?',
                        help='File path to csv file we want to evaluate solution on')
    parser.add_argument('--evalexample', type=str, nargs='?',
                        help='Path to the example file to be matched with csv file')
    parser.add_argument('--training', action='store_true',
                        help='Are we evaluating the dataset on training set only? Uses evalexample as the one it '
                             'should not be evaluated on')
    parser.add_argument('--sol', type=str, nargs='?',
                        help='File path to the solution under evaluation')
    parser.add_argument('--atomisation', action='store_true')
    parser.add_argument('--labs', action='store_true')

    args = parser.parse_args()

    problem_type = ProblemType.ATOMISATION if args.atomisation else ProblemType.GENERALISATION
    loc = "labs" if args.labs else "home"
    base_dir = base_dir_temp[problem_type].format(project_root[loc])
    concepts_file = concepts_file_temp[problem_type].format(base_dir)
    eval_file = concepts_file if args.eval is None else args.eval
    sol_file = solution_file_temp.format(base_dir) if args.sol is None else args.sol
    eval_examples = "" if args.evalexample is None else args.evalexample

    # Generating file paths based on program args
    clingo_out = clingo_out_file_temp.format(base_dir)
    premises_out_dir = premises_out_dir_temp.format(base_dir)
    bk_file = background_knowledge_file_temp.format(base_dir)
    outfile = final_output_temp.format(base_dir)

    p_texts, c_texts = concept_processing.io.load_concept_examples(eval_file)

    # Only consider a subset of lines if evaluating with an example
    lines_eval = ClingoExParser(eval_examples).get_row_ids()
    if lines_eval:
        lines_training = list(set(range(len(p_texts))) - set(lines_eval))
        lines_under_eval = lines_training if args.training else lines_eval
        p_texts = [p_texts[i] for i in lines_under_eval]
        c_texts = [c_texts[i] for i in lines_under_eval]

    spacy_lang = SpacyWrapper()
    create_sent_files(premises_out_dir, p_texts, spacy_lang, problem_type)

    clingo_args = [bk_file, sol_file]
    g_texts = generate_sentences(p_texts, problem_type, premises_out_dir, clingo_args)
    # g_texts = [[add_punctuation(merge_not(truecase(sent, spacy_lang))) for sent in sents] for sents in g_texts]
    g_texts = [[add_punctuation(merge_not(simple_truecase(sent))) for sent in sents] for sents in g_texts]

    save_generalised_sents(p_texts, g_texts, outfile)

    # evalutate

    # Needs to add punctuation because the split removes it for most
    c_texts = [[add_punctuation(x) for x in s.split(". ")] for s in c_texts]

    jaccards = []
    recalls = []
    precisions = []
    for p_text, g_text, c_text in zip(p_texts, g_texts, c_texts):
        j = jaccard(g_text, c_text)
        p = precision(g_text, c_text)
        r = recall(g_text, c_text)
        print(f"Results for '{p_text}' is jaccard:{j}/precision:{p}/recall:{r}")
        jaccards.append(j)
        recalls.append(r)
        precisions.append(p)


    jaccards = np.array(jaccards)
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    std = np.std(jaccards)
    se = std / math.sqrt(jaccards.shape[0])
    print(f"Jaccard: mean {np.mean(jaccards)}, standard deviation {std}, standard error {se} ")
    std = np.std(precisions)
    se = std / math.sqrt(precisions.shape[0])
    print(f"Precision: mean {np.mean(precisions)}, standard deviation {std}, standard error {se} ")
    std = np.std(recalls)
    se = std / math.sqrt(recalls.shape[0])
    print(f"Recall: mean {np.mean(recalls)}, standard deviation {std}, standard error {se} ")
