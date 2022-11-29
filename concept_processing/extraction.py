from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd

from concept_processing.asp.asp_generator import ASPGenerator
from concept_processing.asp.asp_solver import clingo_solve
from concept_processing.asp.clingo_out_parsers import ClingoAnsParser
from concept_processing.enums import ProblemType
from concept_processing.nlp.nlp_parser import NLPParser
from concept_processing.nlp.nlp_utils import add_punctuation, truecase, merge_not

from concept_processing.pam import count_datapoints_in_each_feature

# ILASP solution paths
base_dir = {
    ProblemType.ATOMISATION: '/home/rp218/luke-for-roko/ilasp/atomisation',
    ProblemType.GENERALISATION: '/home/rp218/luke-for-roko/ilasp/generalisation',
}

background_knowledge_file_temp = '{}/background.ilasp'
solution_file_temp = '{}/solutions/best_sol.lp'
clingo_out_file_temp = '{}/clingo_out.tmp'


class ConceptsState:
    def __init__(self, ids: List[str], label_indices: np.ndarray, label_categories: List[str], concept_pam: np.ndarray,
                 concept_strings: List[str]):
        assert len(ids) == len(label_indices) and len(label_indices) == concept_pam.shape[0] and \
               concept_pam.shape[1] == len(concept_strings)
        self.ids = ids
        self.label_indices = label_indices
        self.label_categories = label_categories
        self.concept_pam = concept_pam
        self.concept_strings = concept_strings

    def get_labels(self) -> List[str]:
        return [self.label_categories[i] for i in self.label_indices]

    def to_dict(self) -> Dict[str, List[Any]]:
        return dict(id=self.ids, label=self.get_labels(), concepts=list(self.concept_pam),
                    explanations=self.concept_strings)


# Replaces the old [row_id, [concept_ids]]
class ConceptBag:
    def __init__(self):
        self.store = {}
        self.ids = []
        self.labels = []

    def append(self, row_id: str, concept_ids: List[str], label: str):
        # Two explanations for a video may exist sometimes.
        if row_id not in self.store:
            self.store[row_id] = concept_ids
            self.ids.append(row_id)
            # There is one odd label that needs to be fixed
            if label == ' it could be called a strike because the pitch landed in the strike zone before being hit':
                label = 'strike'

            self.labels.append(label)
        else:
            self.store[row_id] = list(set(self.store[row_id]).union(concept_ids))

    def to_rawbagofconcepts(self) -> List[Tuple[str, List[int]]]:
        return [(id, self.store[id]) for id in self.ids]

    def to_pam(self) -> np.ndarray:
        """
        Creates binary presence-absence matrix (PAM)
        """

        N = len(self.ids)
        C = 0
        for id in self.ids:
            curr_max = np.max(self.store[id], initial=0)
            C = max(C, curr_max)
        C += 1
        data = np.zeros((N, C))
        for i, id_ in enumerate(self.ids):
            data[i, self.store[id_]] = 1

        # Remove extraneous columns
        cols_to_remove = count_datapoints_in_each_feature(data) == 0
        data = data[:, ~cols_to_remove]
        return data


# Applies generalisation/atomisation procedure to extract the concepts
class ConceptExtractor:
    def __init__(self, nlp: NLPParser):
        self.nlp = nlp
        self.concept_dict = {}
        self.next_concept_id = 0
        self.concept_bag = ConceptBag()

    def parse(self, row_id: str, premise_sents: str, label: str):
        # Non need to include errors
        if label != 'none':
            premise_sents = self.nlp(premise_sents)
            premise_sents = [str(sent) for sent in premise_sents.sentences()]

            atomic_sents = self.split(premise_sents, ProblemType.ATOMISATION)
            generalised_sents = self.split(atomic_sents, ProblemType.GENERALISATION)

            concept_ids = [self._get_id(sent) for sent in generalised_sents]
            self.concept_bag.append(row_id, concept_ids, label)

    def _get_id(self, sent: str):
        if sent not in self.concept_dict:
            self.concept_dict[sent] = self.next_concept_id
            self.next_concept_id += 1
        return self.concept_dict[sent]

    def get(self) -> (ConceptBag, List[str]):
        return self.concept_bag, concept_dict_to_list(self.concept_dict)

    @staticmethod
    def _write(clingo_out_file: str, program: List[str]):
        with open(clingo_out_file, 'w') as f:
            for elem in program:
                f.write(elem + '\n')

    def split(self, sents: List[str], problem_type: ProblemType) -> List[str]:

        sols = []
        for sent in sents:
            b_dir = base_dir[problem_type]

            asp_generator = ASPGenerator(self.nlp, problem_type)
            asp_generator.parse(str(sent))
            # Exactly 1 element since we do not have concepts texts
            program = asp_generator.get_programs()[0]

            clingo_out_file = clingo_out_file_temp.format(b_dir)
            solution_file = solution_file_temp.format(b_dir)
            background_file = background_knowledge_file_temp.format(b_dir)

            self._write(clingo_out_file, program)

            atoms = clingo_solve(clingo_out_file, background_file, solution_file)
            asp_parser = ClingoAnsParser(problem_type)

            sents = asp_parser.get_sentences(atoms)
            atomic_sents = [add_punctuation(merge_not(truecase(sent, self.nlp))) for sent in sents]
            sols += atomic_sents
        return sols


def concept_dict_to_list(concept_dict: Dict[str, int]) -> List[str]:
    """
    parameters
    ----------
    concept_dict - dictionary mapping from concept (e.g. strs) to index (int)
        where indices are contiguous and starting from zero.

    returns
    -------
    concepts - a list of concepts where concepts[i] is key k such that
        concept_dict[k] = i
    """
    reverse_dict = {i: s for s, i in concept_dict.items()}
    concepts = [reverse_dict[i] for i in range(len(concept_dict))]
    return concepts
