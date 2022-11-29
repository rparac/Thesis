import pickle

from concept_processing import io
from concept_processing.codex_pipeline import CodexPipeline, extract_concepts
from concept_processing.legacy_code import capture_all_concepts_full_old

# store_location = "/home/rp218/projects/thesis/bird_flowers_ds/"
# store_location = "/home/rp218/projects/thesis/CUB_200_2011/text"
store_location = "/home/rp218/luke-for-roko/full_dataset"
# store_location = "/home/rp218/luke-for-roko/short_dataset"

use_old_pipeline = False


if __name__ == "__main__":
    if use_old_pipeline:
        state_before_grouping = capture_all_concepts_full_old(store_location)
    else:
        original_state = extract_concepts(store_location)

        # state_before_grouping = original_state
        pipeline = CodexPipeline(methods=['simple_pruning'], use_old_pipeline=use_old_pipeline)
        state_before_grouping, simple_pruning_conversion_dict = pipeline(original_state)

    pipeline = CodexPipeline(methods=['grouping', 'pruning'], use_old_pipeline=use_old_pipeline)
    # pipeline = CodexPipeline(methods=['grouping'], use_old_pipeline=use_old_pipeline)

    last_state, conversion_dict_list = pipeline(state_before_grouping)

    csvfname = "grouping_with_new_hyperparameters.csv"
    io.newer_groupings_to_csv(csvfname, state_before_grouping, conversion_dict_list,
                              id_names = ["final_id", "group_id", "start_id"])
                              # id_names = ["group_id", "start_id"])

    # We are using state after simple pruning for hyper-parameters as it is easier to construct matrix labelings that way
    io.store_concept_objects("../jupyter-notebooks/for_hyperparameter_tuning.pkl", state_before_grouping.concept_strings,
                             state_before_grouping.ids, state_before_grouping.label_categories,
                             state_before_grouping.label_indices, state_before_grouping.concept_pam)

    to_store = last_state.to_dict()
    pickle_fname = f"final_dict_{'old' if use_old_pipeline else 'new'}_codex.pkl"
    pickle.dump(to_store, open(pickle_fname, "wb"))
    stored_dict = pickle.load(open(pickle_fname, "rb"))
    print(stored_dict)
    print("Done")
