# Introduction to Machine Learning

## Prerequisites

As described in the instructions, the project is using Python 3.6.7.


## Running the tests

All tests on the code can be executed by running `main.py` on the terminal and passing two arguments, depending on the desired task.

The first argument passed must be the task and it can be:
1. creation (Step 2: Creating Decision Trees)
2. visualization (Bonus Exercise: Visualize Decision Tree)
3. evaluation (Step 3: Evaluation accuracy of created tree on provided dataset)
4. pruning (Step 4: Pruning and evaluation again)

The second argument passed must be the dataset. It can be:
1. clean
2. noisy

*Note:* To run any task on a different dataset, the whole path to the text file must be provided as a second argument

**Example 1**
This example runs the evaluation task on the noisy dataset
```
python3 main.py evaluation noisy
```
**Example 2**
This example runs the pruning task on a different dataset with unknown path to us
```
python3 main.py pruning /path/to/be/provided
```

Both examples assume the tester is in the directory of the project.