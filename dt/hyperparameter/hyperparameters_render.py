from dt.hyperparameter.hyperparameter_selection import HyperparameterSelection


def show_best_params():
    min_samples_split = HyperparameterSelection.get_best_min_samples_split()
    min_samples_leaf = HyperparameterSelection.get_best_min_samples_leaf()
    max_depth = HyperparameterSelection.get_best_max_depth()
    score = HyperparameterSelection.get_best_score()

    print(f'The best min samples split is {min_samples_split} \n'
          f'The best min samples leaf is {min_samples_leaf} \n'
          f'The best max depth is {max_depth} \n'
          f'The best score is {score}')
