import sys
import numpy as np


def get_rnd_gen(seed=None):
    return np.random.default_rng(seed)


def print_update(s):
    sys.stdout.write('\r')
    sys.stdout.write(s)
    sys.stdout.flush()

def get_nn_params_from_file(logfile, load_best_epoch=False, sample_from_iteration=None, seed=None):
    head = next(open(logfile)).split()
    loaded_ws = np.loadtxt(logfile, skiprows=1)
    if load_best_epoch:
        selected_epoch = np.argmax(loaded_ws[:, head.index("running_reward")])
    elif sample_from_iteration is not None:
        rs = get_rnd_gen(seed)
        selected_epoch = rs.integers(-sample_from_iteration, -1)
    else:
        selected_epoch = -1
    print(
        "Selected epoch",
        selected_epoch,
        loaded_ws[:, head.index("reward")][selected_epoch],
        loaded_ws[:, head.index("running_reward")][selected_epoch],
    )
    loadedW = loaded_ws[selected_epoch]
    ind = [head.index(s) for s in head if "coeff_" in s]
    wNN = loadedW[np.min(ind):]
    return wNN


def parse_str(config_entry):
    config_entry = config_entry.split()
    var = []
    for x in config_entry:
        if x == "None":
            var.append(None)
        elif x == "True":
            var.append(True)
        elif x == "False":
            var.append(False)
        else:
            try: var.append(int(x))
            except:
                try: var.append(float(x))
                except:
                    var.append(x)

    if len(var) == 1:
        return var[0]
    else:
        return np.array(var)

