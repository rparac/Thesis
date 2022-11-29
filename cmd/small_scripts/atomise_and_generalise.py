"""
Uses both generalisation and atomisation to get a result for a given sentence.
"""
from concept_processing.codex_pipeline import extract_concepts
from concept_processing.enums import ProblemType
from concept_processing.extraction import ConceptExtractor
from concept_processing.nlp.spacy_wrapper import SpacyWrapper

row_id, label = 'some_id', 'some_label'  # These do not matter for this short script

# The pitch was over the plate but low, missing the strike zone

sentences = ["The pitch was over the plate but low, missing the strike zone"]

nlp = SpacyWrapper()
concept_extractor = ConceptExtractor(nlp)
atomisation_sents = concept_extractor.split(sentences, ProblemType.ATOMISATION)
generalisation_sents = concept_extractor.split(atomisation_sents, ProblemType.GENERALISATION)

print(f"Atomisation sentences are {atomisation_sents}")
print(f"Generalisation sentences are {generalisation_sents}")
