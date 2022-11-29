"""
Finds dep_chains used in generalisation
"""

import concept_processing.io
from concept_processing.asp.asp_solver import clingo_solve
from concept_processing.asp.batch_asp_generator import BatchASPGenerator
from concept_processing.asp.clingo_out_parsers import ClingoAnsParser
from concept_processing.enums import ProblemType
from concept_processing.nlp.spacy_wrapper import SpacyWrapper

examples_file = '/home/rp218/luke-for-roko/ilasp/generalisation/created_examples/examples.csv'
base_dir = '/home/rp218/luke-for-roko/ilasp/generalisation/dep_chain_generation'
sent_file = f'{base_dir}/sent.lp'
logic_file = f'{base_dir}/dep_chain_logic.lp'
out_temp_file = f'{base_dir}/clingo_out.tmp'

if __name__ == "__main__":
    spacy_lang = SpacyWrapper()
    asp_generator = BatchASPGenerator(spacy_lang, ProblemType.GENERALISATION)

    p_texts, _ = concept_processing.io.load_concept_examples(examples_file)

    sols = set()

    asp_generator.parse(p_texts)
    for program in asp_generator.get_programs():
        with open(sent_file, 'w') as f:
            f.write('\n'.join(program))

        out = clingo_solve(sent_file, logic_file)
        ans_parser = ClingoAnsParser(ProblemType.GENERALISATION)
        dep_chains = ans_parser.get_all_dep_chains(out)
        sols |= set(dep_chains)

    print(f"We have {len(sols)} dep chains")
    for i, sol in enumerate(sols):
        print(f"{sol}.")
