from dt.hyperparameter.hyperparameter_selection import HyperparameterSelection
from dt.plot.plots import show_gradient_boost_classifier_lib_impl, show_random_forest_classifier_lib_impl, \
    show_decision_tree_classifier_lib_impl, show_my_decision_tree_impl, show_my_random_forest_impl

min_samples_split = HyperparameterSelection.get_best_min_samples_split()
min_samples_leaf = HyperparameterSelection.get_best_min_samples_leaf()

show_decision_tree_classifier_lib_impl(min_samples_split, min_samples_leaf)
show_random_forest_classifier_lib_impl(min_samples_split, min_samples_leaf)
show_gradient_boost_classifier_lib_impl(min_samples_split, min_samples_leaf)
show_my_decision_tree_impl(min_samples_split, min_samples_leaf)
show_my_random_forest_impl(min_samples_split, min_samples_leaf)
