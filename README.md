# Advent of code 2018

[[**Open the notebook in Colab**]](https://colab.research.google.com/github/hhoppe/advent_of_code_2018/blob/main/advent_of_code_2018.ipynb)

IPython/Jupyter [notebook](https://github.com/hhoppe/advent_of_code_2018/blob/main/advent_of_code_2018.ipynb) by [Hugues Hoppe](http://hhoppe.com/); December 2021.

I participated in the 25-day [Advent of Code](https://adventofcode.com/) for the first time in **2020**.  See [all my notes from back then](https://github.com/hhoppe/advent_of_code_2020).

Here I went back to solve the 2018 puzzles.

This notebook is organized such that each day is self-contained and can be run on its own after the preamble.

- The [**preamble**](#preamble) readies the inputs and answers for the puzzles.  No custom shortcut functions are introduced (other than `check_eq`) so the puzzle code solutions are easily recognizable.

- For **each day**, the first notebook cell defines a `puzzle` object:

  ```
    puzzle = advent.puzzle(day=1)
  ```
  The puzzle input string is automatically read into the attribute `puzzle.input`.
  This input string is unique to each Advent participant.
  By default, the notebook uses [my input data](https://github.com/hhoppe/advent_of_code_2018/tree/main/data) stored on GitHub,
  but the variable `INPUT_PATH_OR_URL_FORMAT` can refer to any URL or
  local file.
  Simlarly, the reference answer to each puzzle part is read using `ANSWER_PATH_OR_URL_FORMAT`.

  Alternatively, we read each puzzle input and answers directly from adventofcode.com using a session cookie and the `advent-of-code-data` PyPI package.

  For each of the two puzzle parts, a function (e.g. `process1`) takes an input string and returns a string or integer answer.
  Using calls like the following, we time the execution of each function and verify the answers:
  ```
    puzzle.verify(part=1, func=process1)
    puzzle.verify(part=2, func=process2)
  ```

- At the end of the notebook, a table summarizes [**timing**](#timings) results.
