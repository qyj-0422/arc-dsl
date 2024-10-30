import os
import json
import inspect
import tqdm

import arc_types
import constants
import dsl
import tests
import solvers
import numpy as np



def get_data(train=True):
    path = f'./data/{"training" if train else "evaluation"}'
    data = {}
    for fn in os.listdir(path):
        with open(f'{path}/{fn}') as f:
            data[fn.rstrip('.json')] = json.load(f)
    ast = lambda g: tuple(tuple(r) for r in g)
    return {
        'train': {k: [{
            'input': ast(e['input']),
            'output': ast(e['output']),
        } for e in v['train']] for k, v in data.items()},
        'test': {k: [{
            'input': ast(e['input']),
            'output': ast(e['output']),
        } for e in v['test']] for k, v in data.items()}
    }


def get_functions(path):
    """ returns a list of available functions """
    with open(path, 'r') as f:
        code = f.read()
    functions = []
    for row in code.split('\n'):
        if row.startswith('def '):
            function = row.split('def ')[1].split('(')[0]
            functions.append(function)
    return functions


def run_dsl_tests(dsl_module, test_module):
    """ test DSL primitives """
    dsl_functions = get_functions(dsl_module.__file__)
    test_functions = get_functions(test_module.__file__)
    expected = set([f'test_{f}' for f in dsl_functions])
    assert set(test_functions) == expected
    for fun in test_functions:
        getattr(test_module, fun)()


def test_solvers_formatting(solvers_module, dsl_module):
    """ tests the implementd solvers for formatting """
    with open('constants.py', 'r') as f:
        constants = [c.split(' = ')[0] for c in f.readlines() if ' = ' in c]
    definitions = {
        function: inspect.getsource(getattr(solvers_module, function)) \
            for function in get_functions(solvers_module.__file__)
    }
    dsl_interface = get_functions(dsl_module.__file__)
    n_correct = 0
    n = len(definitions)
    for key, definition in definitions.items():
        try:
            lines = definition.split('\n')
            assert lines[0] == f'def {key}(I):'
            assert lines[-1] == ''
            variables = set()
            calls = set()
            for line in lines[1:-2]:
                variable, call = line.lstrip().split(' = ')
                function, args = call.split('(')
                assert variable not in dsl_interface
                assert variable not in variables
                assert call not in calls
                variables.add(variable)
                calls.add(call)
                assert function in dsl_interface or function in variables
                assert args[-1] == ')'
                args = [args[:-1]] if ',' not in args else args[:-1].split(', ')
                for arg in args:
                    assert any([
                        arg in variables, arg in dsl_interface,
                        arg in constants, arg == 'I'
                    ])
            for v in variables:
                assert sum([
                    definition.count(vs) for vs in [
                        f'({v})', f'({v}, ', f', {v})',
                        f', {v}, ', f' {v} = ', f' {v}('
                    ]
                ]) > 1 or v == 'O'
            n_correct += 1
        except:
            pass
    print(f'{n_correct} out of {n} solvers formatted correctly.')


def test_solvers_correctness(data, solvers_module):
    """ tests the implemented solvers for correctness """
    n_correct = 0
    n = len(data["train"])
    for key in tqdm.tqdm(data['train'].keys(), total=n):
        task = data['train'][key] + data['test'][key]
        try:
            solver = getattr(solvers_module, f'solve_{key}')
            for ex in task:
                assert solver(ex['input']) == ex['output']
            n_correct += 1
        except:
            pass
    print(f'{n_correct} out of {n} tasks solved correctly.')


def main():
    data = get_data(train=True)
    run_dsl_tests(dsl, tests)
    test_solvers_formatting(solvers, dsl)
    test_solvers_correctness(data, solvers)


# new function added by QYJ
def whether_right_solver(solver, task):
    ast = lambda g: tuple(tuple(r) for r in g)  # 将源格式转为solver所需的格式

    try:
        for ex in task['train']:
            if not solver(ast(ex['input'])) == ast(ex['output']):
                return False
    except:
        return False
    return True


def test_whether_right_solver(solvers_module):
    with open('./data/arc-agi_training_challenges.json', 'r') as f:
        data = json.load(f)
    for n in range(len(data)):
        task = list(data.values())[n]
        t = list(data.keys())[n]
        solver = getattr(solvers_module, f'solve_{t}')
        if not whether_right_solver(solver, task):
            print(f'task {t} is wrong')


def predict_test_output(test_input, solver):
    ast = lambda g: tuple(tuple(r) for r in g)  # 将源格式转为solver所需的格式
    tsa = lambda g: list(list(r) for r in g)  # 将solver所需的格式转为源格式
    return tsa(solver(ast(test_input)))


def test_predict_test_output(solvers_module):
    with open('./data/arc-agi_training_challenges.json', 'r') as f:
        data = json.load(f)
    with open('./data/arc-agi_training_solutions.json', 'r') as f:
        answer = json.load(f)
    for n in range(len(data)):
        task = list(data.values())[n]
        t = list(data.keys())[n]
        solver = getattr(solvers_module, f'solve_{t}')
        for i in range(len(task['test'])):
            test_input = np.array(task['test'][i]['input'])
            try:
                p_answer = np.array(predict_test_output(test_input, solver))
                if not np.array_equal(p_answer, answer[t][i]):
                    print(f'task {t} is wrong')
            except:
                print(f'task {t} can\' be solved')


def find_right_solver_for_task(solvers_module, task):
    functions = get_functions(solvers_module.__file__)
    for func in functions:
        solver = getattr(solvers_module, func)
        if whether_right_solver(solver, task):
            return solver
    return False


def test_find_right_solver_for_task(solvers_module):
    with open('./data/arc-agi_training_challenges.json', 'r') as f:
        data = json.load(f)
    found_num = 0
    same_num = 0
    for n in range(len(data)):
        task = list(data.values())[n]
        t = list(data.keys())[n]
        solver = find_right_solver_for_task(solvers_module, task)
        if solver:
            print(f'solver for task {t} is {solver.__name__}')
            found_num += 1
            if t == solver.__name__:
                same_num += 1
    print(f'{found_num} out of {len(data)} solvers were found correctly.')
    print(f'{same_num} out of {len(data)} solvers were solved by its own solver.')


if __name__ == '__main__':
    main()
    # test_whether_right_solver(solvers)
    # test_predict_test_output(solvers)
    # test_find_right_solver_for_task(solvers)

