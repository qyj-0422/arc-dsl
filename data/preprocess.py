import os
import json

# 获取dataloaders包的路径
__project__ = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(__project__, 'arc-agi_training_challenges.json')
train_solution_path = os.path.join(__project__, 'arc-agi_training_solutions.json')
evaluation_path = os.path.join(__project__, 'arc-agi_evaluation_challenges.json')


def preprocess(train=True):
    data_path = train_path if train else evaluation_path
    target_path = os.path.join(__project__, 'training' if train else 'evaluation')
    with open(data_path) as f:
        data = json.load(f)
    with open(train_solution_path) as f:
        solutions = json.load(f)
    # for k, v in data.items():
    #     with open(os.path.join(target_path, f'{k}.json'), 'w') as f:
    #         json.dump(v, f)
    for k in list(data.keys()):
        v = data[k]
        sol = solutions[k]
        for i in range(len(v['test'])):
            v['test'][i]['output'] = sol[i]
        with open(os.path.join(target_path, f'{k}.json'), 'w') as f:
            json.dump(v, f)


if __name__ == '__main__':
    preprocess()

