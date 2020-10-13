import os
from pathlib import Path

root = Path(
    'snap'
)

task2major = {
    'QQP': 'acc_and_f1',
    'STS-B': 'corr',
    'MRPC': 'acc_and_f1',
}

# The tasks sorted by the amount of data
all_tasks = [
    # 'WNLI',
    'RTE',
    'MRPC',
    'STS-B',
    'CoLA',
    'SST-2',
    'QNLI',
    'QQP',
    'MNLI',
    'MNLI-MM',
]


def print_result(glue_dir):
    print(glue_dir)
    results = {}
    for task in glue_dir.iterdir():
        if task.is_dir():
            eval_fpath = task / 'eval_results.txt'
            task_name = task.name
            if eval_fpath.exists():
                with eval_fpath.open() as f:
                    for line in f:
                        metric, value = line.split('=')
                        metric = metric.strip()
                        value = float(value.strip())
                        if task_name in task2major:
                            if metric == task2major[task_name]:
                                results[task_name] = value
                        else:
                            results[task_name] = value
    if len(results) > 0:
        # sorted_keys = sorted(list(results.keys()))
        # for key in sorted_keys:
        #     print("%8s" % key, end='')
        # print("%8s" % 'GLUE', end='')
        # print()
        # for key in sorted_keys:
        #     print("%8.2f" % (results[key] * 100.), end='')
        # print("%8.2f" % (sum(results.values()) * 100. / len(results)), end='')
        # print()
        for task in all_tasks:
            print("%8s" % task, end='')
        print("%8s" % 'GLUE', end='')
        print()
        for task in all_tasks:
            if task in results:
                result = results[task]
                print("%8.2f" % (result * 100), end='')
            else:
                print(" " * 8, end='')
        mean = lambda x: sum(x) / max(len(x), 1)
        avg_result = mean([value for key, value in results.items() if key in all_tasks])
        print("%8.2f" % (avg_result * 100.), end='')
        print()


def search(path):
    def sorted_key(path):
        try:
            return path.stat().st_mtime
        except Exception:
            return 0.
    path_list = sorted(
        path.iterdir(),
        key=sorted_key
        # x.name
    )
    for subdir in path_list:
        if subdir.is_dir():
            if 'glueepoch_' in subdir.name:
                print_result(subdir)
            else:
                search(subdir)

search(root)
