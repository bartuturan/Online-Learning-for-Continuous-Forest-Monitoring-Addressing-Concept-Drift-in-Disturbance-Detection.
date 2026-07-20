import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight


def build_mlp_model():
    return MLPClassifier(
        hidden_layer_sizes=(64,),
        activation='relu',
        alpha=0.0001,
        random_state=42,
        solver='adam',
        learning_rate='adaptive',
        max_iter=1,
        learning_rate_init=0.001,
        warm_start=False,
        verbose=False,
    )


def compute_year_positive_rate(y_batch, positive_label=1):
    y_batch = np.asarray(y_batch)
    if len(y_batch) == 0:
        return np.nan
    return float(np.mean(y_batch == positive_label))


def compute_binary_class_weights(y_batch, classes=np.array([0, 1]), fallback_mode='smoothed_single_class'):
    y_batch = np.asarray(y_batch, dtype=np.int64)
    if len(y_batch) == 0:
        return {0: 1.0, 1: 1.0}, 'empty_uniform'

    unique_labels = set(np.unique(y_batch).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f'Expected binary labels in {{0, 1}}, got {sorted(unique_labels)}')

    if len(unique_labels) == 2:
        class_weights_array = compute_class_weight('balanced', classes=classes, y=y_batch)
        class_weight_dict = {classes[i]: float(class_weights_array[i]) for i in range(len(classes))}
        return class_weight_dict, 'balanced'

    if fallback_mode != 'smoothed_single_class':
        raise ValueError(f'Unsupported fallback_mode: {fallback_mode}')

    # Laplace-smoothed prevalence keeps weights finite even when one class is absent.
    n_samples = float(len(y_batch))
    positive_count = float(np.sum(y_batch == 1))
    positive_rate_smoothed = (positive_count + 1.0) / (n_samples + 2.0)
    class_weight_1 = 1.0 / (2.0 * positive_rate_smoothed)
    class_weight_0 = 1.0 / (2.0 * (1.0 - positive_rate_smoothed))
    class_weight_dict = {0: float(class_weight_0), 1: float(class_weight_1)}

    present_label = int(next(iter(unique_labels)))
    return class_weight_dict, f'smoothed_single_class_present_{present_label}'


def compute_optimal_f1_threshold(y_true, y_proba, threshold_grid=None, default_threshold=0.5):
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float(default_threshold)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = int(np.argmax(f1_scores))
    if best_idx < len(thresholds):
        return float(thresholds[best_idx])
    return float(default_threshold)


def capture_model_state(model):
    state = {
        'coefs': [w.copy() for w in model.coefs_],
        'intercepts': [b.copy() for b in model.intercepts_],
        'n_layers_': model.n_layers_,
        'n_outputs_': getattr(model, 'n_outputs_', None),
        'out_activation_': getattr(model, 'out_activation_', None),
    }

    optimizer = getattr(model, '_optimizer', None)
    if optimizer is not None and hasattr(optimizer, 'ms'):
        state['optimizer_state'] = {
            'type': 'adam',
            'ms': [m.copy() for m in optimizer.ms],
            'vs': [v.copy() for v in optimizer.vs],
            't': optimizer.t,
        }
    elif optimizer is not None and hasattr(optimizer, 'velocities'):
        state['optimizer_state'] = {
            'type': 'sgd',
            'velocities': [v.copy() for v in optimizer.velocities],
        }
    else:
        state['optimizer_state'] = None

    return state


def restore_model_state(model, state):
    model.coefs_ = [w.copy() for w in state['coefs']]
    model.intercepts_ = [b.copy() for b in state['intercepts']]
    model.n_layers_ = state['n_layers_']
    if state['n_outputs_'] is not None:
        model.n_outputs_ = state['n_outputs_']
    if state['out_activation_'] is not None:
        model.out_activation_ = state['out_activation_']

    opt_state = state.get('optimizer_state')
    if opt_state is not None and hasattr(model, '_optimizer') and model._optimizer is not None:
        opt = model._optimizer
        if opt_state['type'] == 'adam' and hasattr(opt, 'ms'):
            opt.ms = [m.copy() for m in opt_state['ms']]
            opt.vs = [v.copy() for v in opt_state['vs']]
            opt.t = opt_state['t']
        elif opt_state['type'] == 'sgd' and hasattr(opt, 'velocities'):
            opt.velocities = [v.copy() for v in opt_state['velocities']]
