# %% [markdown]
# <a href="https://colab.research.google.com/github/hhoppe/advent_of_code_2018/blob/main/advent_of_code_2018.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Advent of code 2018
#
# [[**Open the notebook in Colab**]](https://colab.research.google.com/github/hhoppe/advent_of_code_2018/blob/main/advent_of_code_2018.ipynb)
#
# IPython/Jupyter [notebook](https://github.com/hhoppe/advent_of_code_2018/blob/main/advent_of_code_2018.ipynb) by [Hugues Hoppe](http://hhoppe.com/) with solutions to the [2018 Advent of Code puzzles](https://adventofcode.com/2018).
# Mostly completed in November 2021.
#
# In this notebook, I explore both "compact" and "fast" code versions, along with data visualizations.
#
# I was able to speed up all the solutions such that the [cumulative time](#timings) across all 25 puzzles is about 7 s.
# (For some puzzles, I had to resort to the `numba` package to jit-compile Python functions.)
#
# Here are some visualization results:
#
# <a href="#day3">day3</a><img src="https://github.com/hhoppe/advent_of_code_2018/raw/main/results/day03.gif" width="200">&emsp;
# <a href="#day6">day6</a><img src="https://github.com/hhoppe/advent_of_code_2018/raw/main/results/day06.gif" width="200">&emsp;
# <a href="#day10">day10</a><img src="https://github.com/hhoppe/advent_of_code_2018/raw/main/results/day10.gif">
# <br/>
# <a href="#day11">day11</a><img src="https://github.com/hhoppe/advent_of_code_2018/raw/main/results/day11.gif" width="200">&emsp;
# <a href="#day12">day12</a><img src="https://github.com/hhoppe/advent_of_code_2018/raw/main/results/day12.png" width="300">&emsp;
# <a href="#day13">day13</a><img src="https://github.com/hhoppe/advent_of_code_2018/raw/main/results/day13.gif" width="200">
# <br/>
# <a href="#day15">day15</a><img src="https://github.com/hhoppe/advent_of_code_2018/raw/main/results/day15a.gif" width="150">
# <img src="https://github.com/hhoppe/advent_of_code_2018/raw/main/results/day15b.gif" width="150">
# <br/>
# <a href="#day17">day17</a><img src="https://github.com/hhoppe/advent_of_code_2018/raw/main/results/day17.png" width="90%">
# <br/>
# <a href="#day18">day18</a><img src="https://github.com/hhoppe/advent_of_code_2018/raw/main/results/day18a.gif" width="200">
# <img src="https://github.com/hhoppe/advent_of_code_2018/raw/main/results/day18b.gif" width="200">&emsp;
# <a href="#day20">day20</a><img src="https://github.com/hhoppe/advent_of_code_2018/raw/main/results/day20.png" width="250">
# <br/>
# <a href="#day22">day22</a><img src="https://github.com/hhoppe/advent_of_code_2018/raw/main/results/day22.gif" width="90%">

# %% [markdown]
# <a name="preamble"></a>
# ## Preamble

# %%
# !command -v ffmpeg >/dev/null || (apt-get -qq update && apt-get -qq -y install ffmpeg) >/dev/null

# %%
# !pip install -q advent-of-code-hhoppe hhoppe-tools mediapy numba

# %%
from __future__ import annotations

import collections
from collections.abc import Callable
import dataclasses
import functools
import heapq
import importlib
import itertools
import math
import re
import textwrap

import advent_of_code_hhoppe  # https://github.com/hhoppe/advent-of-code-hhoppe/blob/main/advent_of_code_hhoppe/__init__.py
import hhoppe_tools as hh  # https://github.com/hhoppe/hhoppe-tools/blob/main/hhoppe_tools/__init__.py
import mediapy as media
import numpy as np
import scipy.optimize

# %%
if not media.video_is_available():
  media.show_videos = lambda *a, **kw: print('Creating video is unavailable.')

# %%
hh.start_timing_notebook_cells()

# %%
YEAR = 2018

# %%
# (1) To obtain puzzle inputs and answers, we first try these paths/URLs:
PROFILE = 'google.Hugues_Hoppe.965276'
# PROFILE = 'github.hhoppe.1452460'
TAR_URL = f'https://github.com/hhoppe/advent_of_code_{YEAR}/raw/main/data/{PROFILE}.tar.gz'
hh.run(f"if [ ! -d data/{PROFILE} ]; then (mkdir -p data && cd data &&"
       f" wget -q {TAR_URL} && tar xzf {PROFILE}.tar.gz); fi")
INPUT_URL = f'data/{PROFILE}/{{year}}_{{day:02d}}_input.txt'
ANSWER_URL = f'data/{PROFILE}/{{year}}_{{day:02d}}{{part_letter}}_answer.txt'

# %%
# (2) If URL not found, we may try adventofcode.com using a session cookie:
if 0:
  # See https://github.com/wimglenn/advent-of-code-data.
  hh.run('rm -f ~/.config/aocd/token*')
  # Fill-in the session cookie in the following:
  hh.run(f"if [ '{PROFILE}' = 'google.Hugues_Hoppe.965276' ]; then mkdir -p ~/.config/aocd && echo 53616... >~/.config/aocd/token; fi")
  hh.run(f"if [ '{PROFILE}' = 'github.hhoppe.1452460' ]; then mkdir -p ~/.config/aocd; echo 53616... >~/.config/aocd/token; fi")
  hh.run('pip install -q advent-of-code-data')
  import aocd

# %%
try:
  import numba
  numba_njit = numba.njit
except ModuleNotFoundError:
  print('Package numba is unavailable.')
  numba_njit = hh.noop_decorator

# %%
advent = advent_of_code_hhoppe.Advent(
    year=YEAR, input_url=INPUT_URL, answer_url=ANSWER_URL)

# %%
hh.adjust_jupyterlab_markdown_width()

# %% [markdown]
# ### Helper functions

# %%
check_eq = hh.check_eq


# %% [markdown]
# ### `Machine` used in several puzzles

# %%
@dataclasses.dataclass
class Machine:
  num_registers: int = 6
  registers: list[int] = dataclasses.field(default_factory=list)
  ip_register: int | None = None
  instructions: list[Machine.Instruction] = dataclasses.field(default_factory=list)
  ip: int = 0

  @dataclasses.dataclass
  class Instruction:
    operation: str
    operands: tuple[int, ...]

  def __post_init__(self) -> None:
    self.registers = [0] * self.num_registers

    def assign(registers: list[int], operands: tuple[int, ...],
               value: int | bool) -> None:
      output = operands[2]
      assert 0 <= output < len(registers)
      registers[output] = int(value)

    self.operations: dict[str, Callable[..., None]] = {
        'addr': lambda r, o: assign(r, o, r[o[0]] + r[o[1]]),
        'addi': lambda r, o: assign(r, o, r[o[0]] + o[1]),
        'mulr': lambda r, o: assign(r, o, r[o[0]] * r[o[1]]),
        'muli': lambda r, o: assign(r, o, r[o[0]] * o[1]),
        'banr': lambda r, o: assign(r, o, r[o[0]] & r[o[1]]),
        'bani': lambda r, o: assign(r, o, r[o[0]] & o[1]),
        'borr': lambda r, o: assign(r, o, r[o[0]] | r[o[1]]),
        'bori': lambda r, o: assign(r, o, r[o[0]] | o[1]),
        'setr': lambda r, o: assign(r, o, r[o[0]]),
        'seti': lambda r, o: assign(r, o, o[0]),
        'gtir': lambda r, o: assign(r, o, o[0] > r[o[1]]),
        'gtri': lambda r, o: assign(r, o, r[o[0]] > o[1]),
        'gtrr': lambda r, o: assign(r, o, r[o[0]] > r[o[1]]),
        'eqir': lambda r, o: assign(r, o, o[0] == r[o[1]]),
        'eqri': lambda r, o: assign(r, o, r[o[0]] == o[1]),
        'eqrr': lambda r, o: assign(r, o, r[o[0]] == r[o[1]]),
    }

  def read_instructions(self, s: str) -> None:
    lines = s.strip('\n').split('\n')
    if lines[0].startswith('#ip'):
      self.ip_register = int(re.fullmatch(r'#ip (\d+)', lines[0]).group(1))
      lines = lines[1:]
    self.instructions = []
    for line in lines:
      operation, *operands = line.split()
      operands2 = tuple(map(int, operands))
      assert operation in self.operations and len(operands2) == 3
      self.instructions.append(self.Instruction(operation, operands2))

  def run_instruction(self, verbose: bool = False) -> None:
    if self.ip_register is not None:
      self.registers[self.ip_register] = self.ip
    instruction = self.instructions[self.ip]
    self.operations[instruction.operation](self.registers, instruction.operands)
    if verbose:
      print(self.ip, instruction.operation, instruction.operands,
            self.registers)
    if self.ip_register is not None:
      self.ip = self.registers[self.ip_register] + 1
    else:
      self.ip += 1


# %% [markdown]
# <a name="day1"></a>
# ## Day 1: Repeat in running sum

# %% [markdown]
# - Part 1: Find sum of list of numbers.
#
# - Part 2: Find value of running sum that first repeats.

# %%
puzzle = advent.puzzle(day=1)


# %%
def process1(s):
  entries = map(int, s.replace(', ', '\n').strip('\n').split('\n'))
  return sum(entries)

check_eq(process1('+1, +1, +1'), 3)
check_eq(process1('+1, +1, -2'), 0)
check_eq(process1('-1, -2, -3'), -6)
puzzle.verify(1, process1)  # ~0 ms.


# %%
def process2(s):
  entries = map(int, s.replace(', ', '\n').strip('\n').split('\n'))
  total = 0
  found = set()
  for value in itertools.cycle(entries):
    found.add(total)
    total += value
    if total in found:
      return total
  assert False

check_eq(process2('+1, -1'), 0)
check_eq(process2('+3, +3, +4, -2, -4'), 10)
check_eq(process2('-6, +3, +8, +5, -6'), 5)
check_eq(process2('+7, +7, -2, -7, -4'), 14)
puzzle.verify(2, process2)  # ~30 ms.

# %% [markdown]
# <a name="day2"></a>
# ## Day 2: Ids with repeated letters

# %% [markdown]
# - Part 1: Count the ids that have a letter repeated twice, and the ids that have a letter repeated thrice.  Return the product of the counts.
#
# - Part 2: Find two ids that differ in just one letter, and return the id without the differing letter.

# %%
puzzle = advent.puzzle(day=2)


# %%
def process1(s):
  sum_twice = sum_thrice = 0
  for id in s.split():
    counts = collections.Counter(id)
    sum_twice += 2 in counts.values()
    sum_thrice += 3 in counts.values()
  return sum_twice * sum_thrice

check_eq(process1('abcdef bababc abbcde abcccd aabcdd abcdee ababab'), 4 * 3)
puzzle.verify(1, process1)  # ~2 ms.


# %%
def process2(s):
  candidates = set()
  for id in s.split():
    for pos in range(len(id)):
      id2 = id[:pos] + '*' + id[pos + 1:]
      if id2 in candidates:
        return id2.replace('*', '')
      candidates.add(id2)
  return None

check_eq(process2('abcde fghij klmno pqrst fguij axcye wvxyz'), 'fgij')
puzzle.verify(2, process2)  # ~4 ms.

# %% [markdown]
# <a name="day3"></a>
# ## Day 3: Overlapping rectangles

# %% [markdown]
# - Part 1: Count the number of grid squares that are covered by at least two rectangles.
#
# - Part 2: Find the rectangle that does not overlap with any other rectangle.

# %%
puzzle = advent.puzzle(day=3)


# %%
def process1(s, part2=False, check_single_solution=False):
  lines = s.strip('\n').split('\n')
  pattern = r'#(\d+) @ (\d+),(\d+): (\d+)x(\d+)'
  grid = collections.defaultdict(int)
  for line in lines:
    claim, l, t, w, h = map(int, re.fullmatch(pattern, line).groups())
    for y in range(t, t + h):
      for x in range(l, l + w):
        grid[y, x] += 1

  if not part2:
    return sum(value >= 2 for value in grid.values())

  found = []
  for line in lines:
    claim, l, t, w, h = map(int, re.fullmatch(pattern, line).groups())
    if all(grid[y, x] == 1 for y in range(t, t + h) for x in range(l, l + w)):
      found.append(claim)
      if not check_single_solution:
        break

  return (lambda x: x)(*found)


check_eq(process1('#1 @ 1,3: 4x4\n#2 @ 3,1: 4x4\n#3 @ 5,5: 2x2'), 4)
puzzle.verify(1, process1)  # ~300 ms.

process2 = functools.partial(process1, part2=True)
check_eq(process2('#1 @ 1,3: 4x4\n#2 @ 3,1: 4x4\n#3 @ 5,5: 2x2'), 3)
puzzle.verify(2, process2)  # ~300 ms.


# %%
def process1(s, part2=False, visualize=False):  # Faster with numpy.
  lines = s.strip('\n').split('\n')
  pattern = r'#(\d+) @ (\d+),(\d+): (\d+)x(\d+)'
  shape = (1000, 1000)
  grid = np.full(shape, 0)
  for line in lines:
    claim, l, t, w, h = map(int, re.fullmatch(pattern, line).groups())
    grid[t: t + h, l: l + w] += 1

  if not part2:
    return np.count_nonzero(grid >= 2)

  claim = None
  for line in lines:
    claim, l, t, w, h = map(int, re.fullmatch(pattern, line).groups())
    if np.all(grid[t: t + h, l: l + w] == 1):
      break

  if visualize:
    image1 = media.to_rgb(grid * 1.0)
    image2 = image1.copy()
    image2[t: t + h, l: l + w] = (0.9, 0.9, 0.0)
    video = [image1, image2]
    shrink = 2
    if shrink > 1:
      shape = (grid.shape[0] // shrink, grid.shape[1] // shrink)
      video = media.resize_video(video, shape)
    media.show_video(video, codec='gif', fps=1)

  return claim


check_eq(process1('#1 @ 1,3: 4x4\n#2 @ 3,1: 4x4\n#3 @ 5,5: 2x2'), 4)
puzzle.verify(1, process1)  # ~13 ms.

process2 = functools.partial(process1, part2=True)
check_eq(process2('#1 @ 1,3: 4x4\n#2 @ 3,1: 4x4\n#3 @ 5,5: 2x2'), 3)
puzzle.verify(2, process2)  # ~21 ms.

# %%
media.set_max_output_height(3000)
_ = process2(puzzle.input, visualize=True)

# %% [markdown]
# <a name="day4"></a>
# ## Day 4: Guard sleep patterns

# %% [markdown]
# - Part 1: Find the guard that has the most minutes asleep.  What minute does that guard spend asleep the most?
#
# - Part 2: Which guard is most frequently asleep on the same minute?

# %%
puzzle = advent.puzzle(day=4)

# %%
s1 = """
[1518-11-01 00:00] Guard #10 begins shift
[1518-11-01 00:05] falls asleep
[1518-11-01 00:25] wakes up
[1518-11-01 00:30] falls asleep
[1518-11-01 00:55] wakes up
[1518-11-01 23:58] Guard #99 begins shift
[1518-11-02 00:40] falls asleep
[1518-11-02 00:50] wakes up
[1518-11-03 00:05] Guard #10 begins shift
[1518-11-03 00:24] falls asleep
[1518-11-03 00:29] wakes up
[1518-11-04 00:02] Guard #99 begins shift
[1518-11-04 00:36] falls asleep
[1518-11-04 00:46] wakes up
[1518-11-05 00:03] Guard #99 begins shift
[1518-11-05 00:45] falls asleep
[1518-11-05 00:55] wakes up
"""


# %%
def process1(s, part2=False):
  lines = s.strip('\n').split('\n')
  lines = sorted(lines)
  num_dates = sum('Guard' in line for line in lines)
  asleep = np.zeros((num_dates, 60))
  date_guard = np.empty(num_dates, dtype=int)
  row = -1
  for line in lines:
    minute, = map(int, re.search(r' \d\d:(\d\d)', line).groups())
    if 'Guard' in line:
      guard, = map(int, re.search(r' Guard #(\d+) begins shift', line).groups())
      row += 1
      date_guard[row] = guard
    elif 'falls asleep' in line:
      asleep_minute = minute
    elif 'wakes up' in line:
      asleep[row][asleep_minute:minute] = 1
    else:
      raise AssertionError()

  guards = set(date_guard)

  if not part2:
    total_sleep = {guard: asleep[date_guard == guard].sum() for guard in guards}
    guard_most_sleep = max(total_sleep, key=total_sleep.get)
    minute_sleep = asleep[date_guard == guard_most_sleep].sum(axis=0)
    minute_most_sleep = minute_sleep.argmax()
    return guard_most_sleep * minute_most_sleep

  guard_sleep_by_minute = {guard: asleep[date_guard == guard].sum(axis=0)
                           for guard in guards}
  guard_max_sleep_by_minute = {
      guard: array.max() for guard, array in guard_sleep_by_minute.items()
  }
  guard = max(guard_max_sleep_by_minute, key=guard_max_sleep_by_minute.get)
  minute = guard_sleep_by_minute[guard].argmax()
  return guard * minute


check_eq(process1(s1), 10 * 24)
puzzle.verify(1, process1)  # ~4 ms.

process2 = functools.partial(process1, part2=True)
check_eq(process2(s1), 99 * 45)
puzzle.verify(2, process2)  # ~4 ms.

# %% [markdown]
# <a name="day5"></a>
# ## Day 5: Polymer string simplification

# %% [markdown]
# - Part 1: Find the length of the string after successively removing all 'bB' and 'Cc' letter pairs.
#
# - Part 2: Do the same after allowing all instances of a single letter to be initially removed.

# %%
puzzle = advent.puzzle(day=5)


# %%
def process1(s, part2=False):  # Slow.

  def simplify_polymer(s):
    pairs = [chr(ord('a') + i) + chr(ord('A') + i) for i in range(26)]
    pairs += [chr(ord('A') + i) + chr(ord('a') + i) for i in range(26)]
    regex = re.compile('|'.join(pairs))
    s_old = None
    while s != s_old:
      s_old = s
      s = regex.sub('', s)
    return s

  def remove_elem(s, i):
    return s.replace(chr(ord('A') + i), '').replace(chr(ord('a') + i), '')

  s = s.strip()
  if not part2:
    return len(simplify_polymer(s))

  return min(len(simplify_polymer(remove_elem(s, i))) for i in range(26))


check_eq(process1('dabAcCaCBAcCcaDA'), 10)
# puzzle.verify(1, process1)  # ~3 s.

process2 = functools.partial(process1, part2=True)
check_eq(process2('dabAcCaCBAcCcaDA'), 4)
# puzzle.verify(2, process2)  # ~80 s.

# %%
def process1(s, part2=False):  # Faster, using stack and numba.

  @numba_njit(cache=True)
  def simplify_polymer(s):
    l = []
    for ch in s:
      if l and abs(ord(ch) - ord(l[-1])) == 32:
        l.pop()
      else:
        l.append(ch)
    return ''.join(l)

  def remove_elem(s, i):
    return s.replace(chr(ord('A') + i), '').replace(chr(ord('a') + i), '')

  s = s.strip()
  if not part2:
    return len(simplify_polymer(s))

  return min(len(simplify_polymer(remove_elem(s, i))) for i in range(26))


check_eq(process1('dabAcCaCBAcCcaDA'), 10)  # ~700 ms for numba compilation.
puzzle.verify(1, process1)  # ~ 20 ms.

process2 = functools.partial(process1, part2=True)
check_eq(process2('dabAcCaCBAcCcaDA'), 4)
puzzle.verify(2, process2)  # ~260 ms with numba (~440 ms without numba)


# %%
def process1(s, part2=False):  # Fastest; nested loops in numba.

  @numba_njit(cache=True)
  def length_of_simplified_polymer(codes):
    l = []
    for code in codes:
      if l and abs(code - l[-1]) == 32:
        l.pop()
      else:
        l.append(code)
    return len(l)

  @numba_njit(cache=True)
  def min_length_after_removing_any_one_letter_pair(codes):
    min_length = 10**8
    for omit in range(26):
      l = []
      for code in codes:
        if code not in (omit, omit + 32):
          if l and abs(code - l[-1]) == 32:
            l.pop()
          else:
            l.append(code)
      min_length = min(min_length, len(l))
    return min_length

  # Convert string to list of codes, each in 0..25 or 32..57 .
  codes = np.array([ord(ch) - ord('A') for ch in s.strip()], dtype=np.int32)
  if not part2:
    return length_of_simplified_polymer(codes)

  return min_length_after_removing_any_one_letter_pair(codes)


check_eq(process1('dabAcCaCBAcCcaDA'), 10)
puzzle.verify(1, process1)  # ~20 ms.

process2 = functools.partial(process1, part2=True)
check_eq(process2('dabAcCaCBAcCcaDA'), 4)
puzzle.verify(2, process2)  # ~43 ms.

# %% [markdown]
# <a name="day6"></a>
# ## Day 6: Voronoi areas

# %% [markdown]
# - Part 1: What is the size of the largest area that isn't infinite?
#
# - Part 2: What is the size of the region containing all locations which have a total distance to all given coordinates of less than 10000?

# %%
puzzle = advent.puzzle(day=6)

# %%
s1 = """
1, 1
1, 6
8, 3
3, 4
5, 5
8, 9
"""


# %%
def process1(s, part2=False, max_sum=10_000, visualize=False):
  yxs = []
  for line in s.strip('\n').split('\n'):
    x, y = map(int, line.split(','))
    yxs.append((y, x))
  shape = np.max(yxs, axis=0) + 1

  def manhattan_from(yx):
    indices = np.indices(shape, dtype=np.int32)
    return abs(indices[0] - yx[0]) + abs(indices[1] - yx[1])

  all_manhattans = np.array([manhattan_from(yx) for yx in yxs])

  if not part2:
    closest = all_manhattans.argmin(axis=0)
    # min_manhattan = all_manhattans.min(axis=0)  # Same but a bit slower.
    min_manhattan = np.take_along_axis(all_manhattans, closest[None], axis=0)[0]
    count_min = np.count_nonzero(all_manhattans == min_manhattan, axis=0)
    closest[count_min > 1] = -1  # Disqualify equidistant locations.
    unbounded = (set(closest[0]) | set(closest[-1]) |
                 set(closest[:, 0]) | set(closest[:, -1]))
    counts = collections.Counter(closest.flat)
    count, i = max((c, i) for i, c in counts.items()
                   if i not in unbounded | {-1})
    if visualize:
      cmap = np.uint8(np.random.default_rng(0).choice(
          range(30, 150), (len(yxs) + 1, 3)))
      image = cmap[closest + 1]
      unb = (closest[..., None] == np.array(list(unbounded))).sum(axis=-1) > 0
      image[unb] += 105
      image[closest == -1] = (255, 255, 255)
      image2 = image.copy()
      image2[closest == i] = (255, 0, 0)
      media.show_video([image, image2], codec='gif', fps=1)
    return count

  sum_manhattans = all_manhattans.sum(axis=0)
  if visualize:
    media.show_image(sum_manhattans < max_sum)
  return np.count_nonzero(sum_manhattans < max_sum)

check_eq(process1(s1), 17)
puzzle.verify(1, process1)  # ~135 ms.
_ = process1(puzzle.input, visualize=True)

# %%
process2 = functools.partial(process1, part2=True)
check_eq(process2(s1, max_sum=32), 16)
puzzle.verify(2, process2)  # ~75 ms.
_ = process2(puzzle.input, visualize=True)

# %% [markdown]
# <a name="day7"></a>
# ## Day 7: Tasks with dependencies

# %% [markdown]
# - Part 1: In what order should the steps in your instructions be completed?
#
# - Part 2: With 5 workers and the 60+ second step durations described above, how long will it take to complete all of the steps?

# %%
puzzle = advent.puzzle(day=7)

# %%
s1 = """
Step C must be finished before step A can begin.
Step C must be finished before step F can begin.
Step A must be finished before step B can begin.
Step A must be finished before step D can begin.
Step B must be finished before step E can begin.
Step D must be finished before step E can begin.
Step F must be finished before step E can begin.
"""


# %%
def process1(s, part2=False, num_workers=5, cost_base=60):
  dependencies = collections.defaultdict(set)
  nodes = set()
  for line in s.strip('\n').split('\n'):
    pattern = r'Step (.) must be finished before step (.) can begin\.'
    node1, node2 = re.fullmatch(pattern, line).groups()
    nodes |= {node1, node2}
    dependencies[node2].add(node1)
  nodelist = sorted(nodes)

  def get_next_node():
    for node in nodelist:
      if not dependencies[node]:
        nodelist.remove(node)
        return node
    return None

  def finish_node(node):
    for set_ in dependencies.values():
      set_ -= {node}

  if not part2:
    result = []
    while nodelist:
      node = get_next_node()
      result.append(node)
      finish_node(node)
    return ''.join(result)

  worker_node = [None] * num_workers
  worker_time = [0] * num_workers
  time = 0
  while True:
    for worker in range(num_workers):
      if worker_node[worker]:
        worker_time[worker] -= 1
        if not worker_time[worker]:
          finish_node(worker_node[worker])
          worker_node[worker] = None
    if not nodelist and not any(worker_node):
      break
    for worker in range(num_workers):
      if not worker_node[worker]:
        node = get_next_node()
        if node:
          worker_node[worker] = node
          worker_time[worker] = cost_base + 1 + ord(node) - ord('A')
    time += 1
  return time


check_eq(process1(s1), 'CABDFE')
puzzle.verify(1, process1)  # ~0 ms.

process2 = functools.partial(process1, part2=True)
check_eq(process2(s1, num_workers=2, cost_base=0), 15)
puzzle.verify(2, process2)  # ~3 ms.

# %% [markdown]
# <a name="day8"></a>
# ## Day 8: Tree from preorder traversal

# %% [markdown]
# - Part 1: What is the sum of all metadata entries?
#
# - Part 2: What is the computed value of the root node?

# %%
puzzle = advent.puzzle(day=8)


# %%
def process1(s, part2=False):

  @dataclasses.dataclass
  class TreeNode:
    children: list[TreeNode]
    metadatas: list[int]

  values = map(int, s.split())

  def parse_tree():
    node = TreeNode([], [])
    num_children = next(values)
    num_metadatas = next(values)
    for _ in range(num_children):
      node.children.append(parse_tree())
    for _ in range(num_metadatas):
      node.metadatas.append(next(values))
    return node

  tree = parse_tree()
  assert next(values, None) is None

  def sum_metadata(node):
    return sum(sum_metadata(n) for n in node.children) + sum(node.metadatas)

  if not part2:
    return sum_metadata(tree)

  def node_value(node):
    if not node.children:
      return sum(node.metadatas)
    return sum(node_value(node.children[child_index - 1])
               for child_index in node.metadatas
               if 1 <= child_index <= len(node.children))

  return node_value(tree)


check_eq(process1('2 3 0 3 10 11 12 1 1 0 1 99 2 1 1 2'), 138)
puzzle.verify(1, process1)  # ~10 ms.

process2 = functools.partial(process1, part2=True)
check_eq(process2('2 3 0 3 10 11 12 1 1 0 1 99 2 1 1 2'), 66)
puzzle.verify(2, process2)  # ~10 ms.

# %% [markdown]
# <a name="day9"></a>
# ## Day 9: Circle of marbles

# %% [markdown]
# - Part 1: What is the winning Elf's score?
#
# - Part 2: What would the new winning Elf's score be if the number of the last marble were 100 times larger?

# %%
puzzle = advent.puzzle(day=9)


# %%
def process1(s, part2=False):  # Compact.
  pattern = r'(\d+) players; last marble is worth (\d+) points'
  num_players, last_marble = map(int, re.fullmatch(pattern, s.strip()).groups())
  if part2:
    last_marble *= 100
  marbles = collections.deque([0])
  scores = [0] * num_players
  for marble in range(1, last_marble + 1):
    if marble % 23 == 0:
      marbles.rotate(7)
      scores[marble % num_players] += marble + marbles.popleft()
    else:
      marbles.rotate(-2)
      marbles.appendleft(marble)

  return max(scores)


check_eq(process1('10 players; last marble is worth 1618 points'), 8317)
check_eq(process1('13 players; last marble is worth 7999 points'), 146373)
check_eq(process1('17 players; last marble is worth 1104 points'), 2764)
check_eq(process1('21 players; last marble is worth 6111 points'), 54718)
check_eq(process1('30 players; last marble is worth 5807 points'), 37305)
puzzle.verify(1, process1)  # ~20 ms.

process2 = functools.partial(process1, part2=True)
# puzzle.verify(2, process2)  # ~2000 ms.

# %%
def process1(s, part2=False):  # Slightly faster with quick inner loop.
  pattern = r'(\d+) players; last marble is worth (\d+) points'
  num_players, last_marble = map(int, re.fullmatch(pattern, s.strip()).groups())
  if part2:
    last_marble *= 100
  # Note that numba does not support deque; there is a feature request:
  # https://githubmemory.com/repo/numba/numba/issues/7417
  marbles = collections.deque([0])
  scores = [0] * num_players
  marble = 1
  while marble < last_marble - 23:
    for marble in range(marble, marble + 22):
      marbles.rotate(-1)
      marbles.append(marble)
    marble += 1
    marbles.rotate(7)
    scores[marble % num_players] += marble + marbles.pop()
    marbles.rotate(-1)
    marble += 1

  for marble in range(marble, last_marble + 1):
    if marble % 23 == 0:
      marbles.rotate(7)
      scores[marble % num_players] += marble + marbles.pop()
      marbles.rotate(-1)
    else:
      marbles.rotate(-1)
      marbles.append(marble)

  return max(scores)


check_eq(process1('10 players; last marble is worth 1618 points'), 8317)
check_eq(process1('13 players; last marble is worth 7999 points'), 146373)
check_eq(process1('17 players; last marble is worth 1104 points'), 2764)
check_eq(process1('21 players; last marble is worth 6111 points'), 54718)
check_eq(process1('30 players; last marble is worth 5807 points'), 37305)
puzzle.verify(1, process1)  # ~17 ms.

process2 = functools.partial(process1, part2=True)
# puzzle.verify(2, process2)  # ~1800 ms.

# %%
def process1(s, part2=False):  # Fastest.  Singly-linked list is sufficient!
  pattern = r'(\d+) players; last marble is worth (\d+) points'
  num_players, last_marble = map(int, re.fullmatch(pattern, s.strip()).groups())
  if part2:
    last_marble *= 100

  @numba_njit(cache=True)
  def func(num_players, last_marble):
    scores = [0] * num_players
    # "array[marble1] == marble2" indicates that marble2 is next after marble1.
    array = np.empty(last_marble + 23, dtype=np.int32)
    array[0] = 1
    array[1] = 0
    marble = 2

    while True:
      for marble in range(marble, marble + 21):  # e.g., [2, ..., 22]
        marble1 = array[marble - 1]
        marble2 = array[marble1]
        array[marble1] = marble
        array[marble] = marble2
      marble += 1  # e.g., 23
      popped = array[marble - 5]  # e.g., 9 = next(18)
      if marble > last_marble:
        break
      scores[marble % num_players] += marble + popped
      array[marble - 5] = marble - 4  # Remove popped.
      next19 = array[marble - 4]
      array[marble + 1] = array[next19]  # e.g., next(24) = next(next(19))
      array[next19] = marble + 1  # e.g., next(next(19)) = 24
      marble += 2  # e.g., 25 == 2 (mod 23)

    return max(scores)

  return func(num_players, last_marble)


check_eq(process1('10 players; last marble is worth 1618 points'), 8317)
check_eq(process1('13 players; last marble is worth 7999 points'), 146373)
check_eq(process1('17 players; last marble is worth 1104 points'), 2764)
check_eq(process1('21 players; last marble is worth 6111 points'), 54718)
check_eq(process1('30 players; last marble is worth 5807 points'), 37305)
puzzle.verify(1, process1)  # ~6 ms  (~50 ms without numba)

process2 = functools.partial(process1, part2=True)
puzzle.verify(2, process2)  # ~38 ms (~5100 ms without numba)

# %% [markdown]
# <a name="day10"></a>
# ## Day 10: Message from moving points

# %% [markdown]
# - Part 1: What message will eventually appear in the sky?
#
# - Part 2: how many seconds would they have needed to wait for that message to appear?

# %%
puzzle = advent.puzzle(day=10)

# %%
s1 = """
position=< 9,  1> velocity=< 0,  2>
position=< 7,  0> velocity=<-1,  0>
position=< 3, -2> velocity=<-1,  1>
position=< 6, 10> velocity=<-2, -1>
position=< 2, -4> velocity=< 2,  2>
position=<-6, 10> velocity=< 2, -2>
position=< 1,  8> velocity=< 1, -1>
position=< 1,  7> velocity=< 1,  0>
position=<-3, 11> velocity=< 1, -2>
position=< 7,  6> velocity=<-1, -1>
position=<-2,  3> velocity=< 1,  0>
position=<-4,  3> velocity=< 2,  0>
position=<10, -3> velocity=<-1,  1>
position=< 5, 11> velocity=< 1, -2>
position=< 4,  7> velocity=< 0, -1>
position=< 8, -2> velocity=< 0,  1>
position=<15,  0> velocity=<-2,  0>
position=< 1,  6> velocity=< 1,  0>
position=< 8,  9> velocity=< 0, -1>
position=< 3,  3> velocity=<-1,  1>
position=< 0,  5> velocity=< 0, -1>
position=<-2,  2> velocity=< 2,  0>
position=< 5, -2> velocity=< 1,  2>
position=< 1,  4> velocity=< 2,  1>
position=<-2,  7> velocity=< 2, -2>
position=< 3,  6> velocity=<-1, -1>
position=< 5,  0> velocity=< 1,  0>
position=<-6,  0> velocity=< 2,  0>
position=< 5,  9> velocity=< 1, -2>
position=<14,  7> velocity=<-2,  0>
position=<-3,  6> velocity=< 2, -1>
"""

# %%
# "Support for 10-pixel-tall characters (2018 Day 10) is coming soon."
if 0:  # https://pypi.org/project/advent-of-code-ocr/
  hh.run('!pip -q install advent-of-code-ocr')
  import advent_of_code_ocr


# %%
def process1(s, part2=False):  # Slow.
  positions, velocities = [], []
  for line in s.strip('\n').split('\n'):
    pattern = r'position=< *(\S+), *(\S+)> velocity=< *(\S+), *(\S+)>'
    x, y, dx, dy = map(int, re.fullmatch(pattern, line).groups())
    positions.append([y, x])
    velocities.append([dy, dx])
  positions, velocities = np.array(positions), np.array(velocities)

  for index in itertools.count():
    shape = positions.ptp(axis=0) + 1
    if shape[0] == 8 or (shape[0] == 10 and shape[1] > 30):
      break
    positions += velocities

  if part2:
    return index

  grid = hh.grid_from_indices(positions, dtype=np.uint8)
  import hashlib
  hashed = hashlib.md5(''.join(map(str, grid.flat)).encode()).hexdigest()
  # print(hashed)
  return {
      '05d005c2fd38c74568ab697305825ff6': 'FPRBRRZA',  # google.Hugues_Hoppe.965276
      '7a115ac723c75059c742c8bb21d5ee1c': 'ERCXLAJL',  # github.hhoppe.1452460
  }[hashed] if shape[1] > 30 else 'HI'

check_eq(process1(s1), 'HI')
puzzle.verify(1, process1)  # ~230 ms.

process2 = functools.partial(process1, part2=True)
check_eq(process2(s1), 3)
puzzle.verify(2, process2)  # ~230 ms.


# %%
def process1(s, part2=False, visualize=False):  # Quick initial jump; visualize.
  positions, velocities = [], []
  for line in s.strip('\n').split('\n'):
    pattern = r'position=< *(\S+), *(\S+)> velocity=< *(\S+), *(\S+)>'
    x, y, dx, dy = map(int, re.fullmatch(pattern, line).groups())
    positions.append([y, x])
    velocities.append([dy, dx])
  positions, velocities = np.array(positions), np.array(velocities)
  all_positions = []

  leftmost, rightmost = positions[:, 1].argmin(), positions[:, 1].argmax()
  index = max(
      (positions[rightmost, 1] - positions[leftmost, 1]) // abs(
          velocities[leftmost, 1] - velocities[rightmost, 1]) - 20, 0)
  positions += velocities * index

  while True:
    if visualize:
      all_positions.extend([(index, y, x) for y, x in positions])
    shape = positions.ptp(axis=0) + 1
    if shape[0] == 8 or (shape[0] == 10 and shape[1] > 30):
      break
    positions += velocities
    index += 1

  if part2:
    return index

  grid = hh.grid_from_indices(positions, dtype=np.uint8)
  if visualize:
    media.show_image(np.pad(grid, 1), height=50, border=True)
    video = hh.grid_from_indices(all_positions, dtype=float)
    video = [video[0]] * 5 + list(video) + [video[-1]] * 10
    media.show_video(video, codec='gif', fps=5)
  import hashlib
  hashed = hashlib.md5(''.join(map(str, grid.flat)).encode()).hexdigest()
  # print(hashed)
  return {
      '05d005c2fd38c74568ab697305825ff6': 'FPRBRRZA',  # google.Hugues_Hoppe.965276
      '7a115ac723c75059c742c8bb21d5ee1c': 'ERCXLAJL',  # github.hhoppe.1452460
  }[hashed] if shape[1] > 30 else 'HI'

check_eq(process1(s1), 'HI')
puzzle.verify(1, process1)  # ~3 ms.

process2 = functools.partial(process1, part2=True)
check_eq(process2(s1), 3)
puzzle.verify(2, process2)  # ~3 ms.

_ = process1(puzzle.input, visualize=True)

# %% [markdown]
# <a name="day11"></a>
# ## Day 11: Grid square with largest sum

# %% [markdown]
# - Part 1: What is the `X,Y` coordinate of the top-left fuel cell of the 3x3 square with the largest total power?
#
# - Part 2: What is the `X,Y,size` identifier of the square with the largest total power?

# %%
puzzle = advent.puzzle(day=11)


# %%
def process1(s, part2=False, visualize=False):

  def power_level(yx, serial_number):
    rack_id = yx[1] + 10
    return ((rack_id * yx[0] + serial_number) * rack_id // 100) % 10 - 5

  check_eq(power_level([5, 3], serial_number=8), 4)
  check_eq(power_level([79, 122], serial_number=57), -5)
  check_eq(power_level([196, 217], serial_number=39), 0)
  check_eq(power_level([153, 101], serial_number=71), 4)

  serial_number = int(s)
  indices = np.indices((300, 300), np.int32)
  power = power_level(indices + 1, serial_number=serial_number)
  integral = np.pad(power, [[1, 0], [1, 0]]).cumsum(axis=0).cumsum(axis=1)

  def get_yx_largest(size):
    if 0:  # Slower.
      import scipy.signal
      box = np.full((size, size), 1, dtype=np.int32)
      result = scipy.signal.convolve2d(power, box, mode='valid')
    else:
      result = (integral[size:, size:] - integral[size:, :-size]
                - integral[:-size, size:] + integral[:-size, :-size])
    yx_largest = np.unravel_index(result.argmax(), result.shape)
    return yx_largest, result[yx_largest]

  if not part2:
    yx_largest = get_yx_largest(size=3)[0]
    return f'{yx_largest[1] + 1},{yx_largest[0] + 1}'

  results = {size: get_yx_largest(size) for size in range(1, 301)}
  best_size = max(results, key=lambda size: results[size][1])
  yx_largest, _ = results[best_size]

  if visualize:
    image = media.to_rgb(power * 1.0, cmap='bwr')
    image2 = image.copy()
    image2[yx_largest[0]: yx_largest[0] + best_size,
           yx_largest[1]: yx_largest[1] + best_size] *= 0.7
    image = image.repeat(2, axis=0).repeat(2, axis=1)
    image2 = image2.repeat(2, axis=0).repeat(2, axis=1)
    media.show_video([image, image2], codec='gif', fps=1)

  return f'{yx_largest[1] + 1},{yx_largest[0] + 1},{best_size}'


check_eq(process1('18'), '33,45')
check_eq(process1('42'), '21,61')
puzzle.verify(1, process1)  # ~3 ms.

process2 = functools.partial(process1, part2=True)
check_eq(process2('18'), '90,269,16')
check_eq(process2('42'), '232,251,12')
puzzle.verify(2, process2)  # ~70 ms.
_ = process2(puzzle.input, visualize=True)

# %% [markdown]
# The blinking (darkened) region at the lower-right indicates the square
# with the largest sum of values in the image.
# (Blue indicates small values; red indicates large values.)

# %% [markdown]
# <a name="day12"></a>
# ## Day 12: Cellular automaton in 1D

# %% [markdown]
# - Part 1: After 20 generations, what is the sum of the numbers of all pots which contain a plant?
#
# - Part 2: After fifty billion (50000000000) generations, what is the sum of the numbers of all pots which contain a plant?

# %%
puzzle = advent.puzzle(day=12)

# %%
s1 = """
initial state: #..#.#..##......###...###

...## => #
..#.. => #
.#... => #
.#.#. => #
.#.## => #
.##.. => #
.#### => #
#.#.# => #
#.### => #
##.#. => #
##.## => #
###.. => #
###.# => #
####. => #
"""


# %%
def process1(s, part2=False, visualize=False):
  lines = s.strip('\n').split('\n')
  assert lines[0].startswith('initial state: ') and not lines[1]
  state = lines[0][15:]
  rules = {''.join(key): '.' for key in itertools.product('.#', repeat=5)}
  for line in lines[2:]:
    key, value = line.split(' => ')
    rules[key] = value
  states = [state]

  def sum_pots(state, pad_len):
    return sum(i - pad_len for i, ch in enumerate(state) if ch == '#')

  for index in range(10**8 if part2 else 20):
    # Grown the domain by two extra '.' on both sides.
    state = '....' + state + '....'
    state = ''.join(rules[''.join(w)] for w in hh.sliding_window(state, 5))
    states.append(state)
    if part2 and len(states) >= 2:
      if any(states[-2] == states[-1][2 + shift: 2 + shift + len(states[-2])]
             for shift in (-1, 0, 1)):
        break

  if visualize:
    padded_states = []
    for index, state in enumerate(states):
      pad_len = (len(states) - 1 - index) * 2
      padded_states.append('.' * pad_len + state + '.' * pad_len)
    grid = np.array([list(state) for state in padded_states]) == '#'
    xmin, xmax = np.nonzero(grid.any(axis=0))[0][[0, -1]]
    grid = grid[:, max(xmin - 4, 0):xmax + 5]
    grid = grid.repeat(2, axis=0).repeat(2, axis=1)
    media.show_image(grid)

  if not part2:
    return sum_pots(states[-1], pad_len=2 * (len(states) - 1))

  sum_pots2 = sum_pots(states[-1], pad_len=2 * (len(states) - 1))
  sum_pots1 = sum_pots(states[-2], pad_len=2 * (len(states) - 2))
  diff_sum_pots = sum_pots2 - sum_pots1
  remaining_generations = 50000000000 - (len(states) - 1)
  return sum_pots2 + diff_sum_pots * remaining_generations


check_eq(process1(s1), 325)
puzzle.verify(1, process1)  # ~2 ms.

process2 = functools.partial(process1, part2=True)
puzzle.verify(2, process2)  # ~20 ms.
_ = process2(puzzle.input, visualize=True)

# %% [markdown]
# For my input, after 102 generations, the output has converged to a constant pattern that is translating rightward by 1 space every generation.
#

# %% [markdown]
# <a name="day13"></a>
# ## Day 13: Carts on tracks

# %% [markdown]
# - Part 1: What is the `x,y` location of the first crash?
#
# - Part 2: What is the location of the last cart at the end of the first tick where it is the only cart left?

# %%
puzzle = advent.puzzle(day=13)

# %%
s1 = r"""
/->-\        EOL
|   |  /----\
| /-+--+-\  |
| | |  | v  |
\-+-/  \-+--/
  \------/   EOL
""".replace('EOL', '')

s2 = r"""
/>-<\  EOL
|   |  EOL
| /<+-\
| | | v
\>+</ |
  |   ^
  \<->/
""".replace('EOL', '')


# %%
def process1(s, part2=False, verbose=False, visualize=False):

  @dataclasses.dataclass
  class Cart:
    yx: tuple[int, int]
    direction: str  # '<', '>', 'v', or '^'.
    next_turn: int = 0  # Three states: 0=left, 1=straight, 2=right.

  grid = hh.grid_from_string(s)
  carts = []
  for ch in '<>v^':
    for yx in zip(*np.nonzero(grid == ch)):
      grid[yx] = {'<': '-', '>': '-', 'v': '|', '^': '|'}[ch]
      carts.append(Cart(yx, ch))

  def text_from_grid():
    grid2 = grid.copy()
    for cart in carts:
      grid2[cart.yx] = cart.direction
    return hh.string_from_grid(grid2)

  check_eq(text_from_grid(), s.strip('\n'))

  if visualize:
    cmap = {' ': (250,) * 3, '+': (140, 140, 140),
            **{ch: (180,) * 3 for ch in r'|-\/'}}
    image0 = np.array(
      [cmap[e] for e in grid.flat], dtype=np.uint8).reshape(*grid.shape, 3)
  images = []
  for iteration in itertools.count():
    assert len(carts) >= 2
    if visualize:
      image = image0.copy()
      for cart in carts:
        image[cart.yx] = (0, 100, 0)
      images.append(image)
    if verbose:
      print(text_from_grid())
    for cart in sorted(carts, key=lambda cart: cart.yx):
      if cart.yx[0] == -1:
        continue
      dyx = {'<': (0, -1), '>': (0, 1),
             '^': (-1, 0), 'v': (1, 0)}[cart.direction]
      # new_yx = tuple(np.array(cart.yx) + dyx)
      new_yx = cart.yx[0] + dyx[0], cart.yx[1] + dyx[1]
      cart2 = next((cart2 for cart2 in carts if cart2.yx == new_yx), None)
      if cart2:  # Collison.
        if not part2:
          if visualize:
            image = image.copy()
            image[new_yx] = (255, 0, 0)
            images.append(image)
            images = [im.repeat(3, axis=0).repeat(3, axis=1) for im in images]
            images = [images[0]] * 20 + images + [images[-1]] * 40
            media.show_video(images, codec='gif', fps=20)
            return
          print(f'first collision at iteration={iteration}')
          return f'{new_yx[1]},{new_yx[0]}'
        for crashed_cart in [cart, cart2]:
          crashed_cart.yx = (-1, -1)
        continue
      ch = grid[new_yx]
      assert ch in '/\\+-|', ord(ch)
      if ch in '/\\':
        cart.direction = {
            '</': 'v', '<\\': '^', '>/': '^', '>\\': 'v',
            '^/': '>', '^\\': '<', 'v/': '<', 'v\\': '>',
        }[cart.direction + ch]
      elif ch == '+':
        if cart.next_turn in (0, 2):
          cart.direction = {
              '<0': 'v', '<2': '^', '>0': '^', '>2': 'v',
              '^0': '<', '^2': '>', 'v0': '>', 'v2': '<',
          }[cart.direction + str(cart.next_turn)]
        cart.next_turn = (cart.next_turn + 1) % 3
      cart.yx = new_yx
    carts = [cart for cart in carts if cart.yx[0] != -1]
    if part2 and len(carts) == 1:
      print(f'Only one cart left at iteration={iteration}')
      return f'{carts[0].yx[1]},{carts[0].yx[0]}'


check_eq(process1(s1), '7,3')
puzzle.verify(1, process1)  # ~19 ms.

# %%
_ = process1(puzzle.input, visualize=True)  # Slow; ~2.5 s.

# %%
process2 = functools.partial(process1, part2=True)
check_eq(process2(s2), '6,4')
puzzle.verify(2, process2)  # ~280 ms.

# %% [markdown]
# <a name="day14"></a>
# ## Day 14: Combining recipes

# %% [markdown]
# - Part 1: What are the scores of the ten recipes immediately after the number of recipes in your puzzle input?
#
# - Part 2: How many recipes appear on the scoreboard to the left of the score sequence in your puzzle input?

# %%
puzzle = advent.puzzle(day=14)


# %%
def process1(s):  # Slow.
  num_recipes = int(s)
  recipes = [3, 7]
  indices = [0, 1]

  while len(recipes) < num_recipes + 10:
    currents = [recipes[index] for index in indices]
    total = sum(currents)
    digits = [1, total - 10] if total >= 10 else [total]
    recipes.extend(digits)
    indices = [(index + 1 + recipes[index]) % len(recipes) for index in indices]

  return ''.join(map(str, recipes[num_recipes:num_recipes + 10]))


check_eq(process1('9'), '5158916779')
check_eq(process1('5'), '0124515891')
check_eq(process1('18'), '9251071085')
check_eq(process1('2018'), '5941429882')
puzzle.verify(1, process1)  # ~820 ms.


# %%
def process1(s):  # Fast using numba.
  num_recipes = int(s)

  @numba_njit(cache=True)
  def func(num_recipes):
    recipes = np.full(num_recipes + 11, 1, dtype=np.uint8)
    num = 2
    recipes[0] = 3
    recipes[1] = 7
    index0, index1 = 0, 1
    while num < num_recipes + 10:
      current0, current1 = recipes[index0], recipes[index1]
      total = current0 + current1
      if total >= 10:
        recipes[num + 1] = total - 10
        num += 2
      else:
        recipes[num] = total
        num += 1
      index0 += 1 + current0
      if index0 >= num:
        index0 %= num
      index1 += 1 + current1
      if index1 >= num:
        index1 %= num
    return recipes[num_recipes:num_recipes + 10]

  return ''.join(map(str, func(num_recipes)))


check_eq(process1('9'), '5158916779')
check_eq(process1('5'), '0124515891')
check_eq(process1('18'), '9251071085')
check_eq(process1('2018'), '5941429882')
puzzle.verify(1, process1)  # ~8 ms.


# %%
def process2(s):  # Slow.
  pattern = list(map(int, s.strip()))
  recipes = [3, 7]
  indices = [0, 1]

  while True:
    currents = [recipes[index] for index in indices]
    total = sum(currents)
    digits = [1, total - 10] if total >= 10 else [total]
    for digit in digits:
      recipes.append(digit)
      if recipes[-len(pattern):] == pattern:
        return len(recipes) - len(pattern)
    indices = [(index + 1 + recipes[index]) % len(recipes) for index in indices]

check_eq(process2('92510'), 18)
check_eq(process2('59414'), 2018)
# puzzle.verify(2, process2)  # ~30 s.

# %%
def process2(s):  # Fast using numba.
  pattern = np.array([int(ch) for ch in s.strip()], dtype=np.uint8)

  @numba_njit(cache=True)
  def func(pattern):
    len_pattern = len(pattern)
    max_recipes = 100_000_000
    recipes = np.empty(max_recipes, dtype=np.uint8)
    num = 2
    recipes[0] = 3
    recipes[1] = 7
    index0, index1 = 0, 1

    def matches():
      if num < len_pattern:
        return False
      for i in range(len_pattern):
        if recipes[num - len_pattern + i] != pattern[i]:
          return False
      return True

    while True:
      assert num + 2 < max_recipes
      current0, current1 = recipes[index0], recipes[index1]
      total = current0 + current1
      if total >= 10:
        recipes[num] = 1
        num += 1
        if matches():
          break
        recipes[num] = total - 10
        num += 1
        if matches():
          break
      else:
        recipes[num] = total
        num += 1
        if matches():
          break
      index0 = (index0 + 1 + current0) % num
      index1 = (index1 + 1 + current1) % num
    return num - len_pattern

  return func(pattern)


check_eq(process2('51589'), 9)
check_eq(process2('01245'), 5)
check_eq(process2('92510'), 18)
check_eq(process2('59414'), 2018)
puzzle.verify(2, process2)  # ~360 ms.


# %%
def process2(s):  # Faster by generating batches.
  pattern = np.array([int(ch) for ch in s.strip()], dtype=np.uint8)

  @numba_njit(cache=True)
  def func(pattern):
    max_recipes = 100_000_000
    recipes = np.empty(max_recipes, dtype=np.uint8)
    num = 2
    recipes[0] = 3
    recipes[1] = 7
    index0, index1 = 0, 1
    batch_size = 1000

    while True:
      assert num + batch_size * 2 < max_recipes
      prev_num = max(num - len(pattern) + 1, 0)

      # Generate batch.
      for _ in range(batch_size):
        current0, current1 = recipes[index0], recipes[index1]
        total = current0 + current1
        if total >= 10:
          recipes[num] = 1
          recipes[num + 1] = total - 10
          num += 2
        else:
          recipes[num] = total
          num += 1
        index0 += 1 + current0
        if index0 >= num:
          index0 %= num
        index1 += 1 + current1
        if index1 >= num:
          index1 %= num

      # Find pattern in new batch results.
      sequence = recipes[prev_num:num]
      n = len(sequence)
      m = len(pattern)
      for i in range(n - m + 1):
        for j in range(m):
          if sequence[i + j] != pattern[j]:
            break
        else:
          return prev_num + i

  return func(pattern)


check_eq(process2('51589'), 9)
check_eq(process2('01245'), 5)
check_eq(process2('92510'), 18)
check_eq(process2('59414'), 2018)
puzzle.verify(2, process2)  # ~190 ms.


# %%
def process2(s):  # Try using Knuth-Morris-Pratt (KMP); not a win for 6-subseq.
  pattern = np.array([int(ch) for ch in s.strip()], dtype=np.uint8)

  @numba_njit(cache=True)
  def func(pattern):

    # Precompute offsets for Knuth-Morris-Pratt (KMP) subsequence search; see
    # https://www.py4u.net/discuss/12693.
    def kmp_offsets(subseq):
      m = len(subseq)
      offsets = np.zeros(m, dtype=np.int64)
      j = 1
      k = 0
      while j < m:
        if subseq[j] == subseq[k]:
          k += 1
          offsets[j] = k
          j += 1
        else:
          if k != 0:
            k = offsets[k - 1]
          else:
            offsets[j] = 0
            j += 1
      return offsets

    max_recipes = 100_000_000
    recipes = np.empty(max_recipes, dtype=np.uint8)
    num = 2
    recipes[0] = 3
    recipes[1] = 7
    index0, index1 = 0, 1
    batch_size = 1000
    offsets = kmp_offsets(pattern)

    while True:
      assert num + batch_size * 2 < max_recipes
      prev_num = max(num - len(pattern) + 1, 0)

      # Generate batch.
      for _ in range(batch_size):
        current0, current1 = recipes[index0], recipes[index1]
        total = current0 + current1
        if total >= 10:
          recipes[num] = 1
          recipes[num + 1] = total - 10
          num += 2
        else:
          recipes[num] = total
          num += 1
        index0 += 1 + current0
        if index0 >= num:
          index0 %= num
        index1 += 1 + current1
        if index1 >= num:
          index1 %= num

      # Find pattern in new batch results using KMP.
      seq = recipes[prev_num:num]
      subseq = pattern
      m = len(subseq)
      n = len(seq)
      i = j = 0
      while i < n:
        if seq[i] == subseq[j]:
          i += 1
          j += 1
        if j == m:
          return prev_num + (i - j)
        if i < n and seq[i] != subseq[j]:
          if j != 0:
            j = offsets[j - 1]
          else:
            i += 1

  return func(pattern)


check_eq(process2('51589'), 9)
check_eq(process2('01245'), 5)
check_eq(process2('92510'), 18)
check_eq(process2('59414'), 2018)
puzzle.verify(2, process2)  # ~250 ms is slower than naive algorithm.


# %%
def process2(s):  # Fastest, using Boyer-Moore-Horspool subsequence search.
  pattern = np.array([int(ch) for ch in s.strip()], dtype=np.uint8)

  @numba_njit(cache=True)
  def func(pattern):

    # Precompute skips for Boyer-Moore-Horspool algorithm; see
    # https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore%E2%80%93Horspool_algorithm
    def boyer_moore_horspool_skip_table(subseq, alphabet_size):
      table = np.full(alphabet_size, len(subseq))
      for i in range(len(subseq) - 1):
        table[subseq[i]] = len(subseq) - 1 - i
      return table

    max_recipes = 100_000_000
    recipes = np.empty(max_recipes, dtype=np.uint8)
    num = 2
    recipes[0] = 3
    recipes[1] = 7
    index0, index1 = 0, 1
    batch_size = 1000
    skip_table = boyer_moore_horspool_skip_table(pattern, 10)

    while True:
      assert num + batch_size * 2 < max_recipes
      prev_num = max(num - len(pattern) + 1, 0)

      # Generate batch.
      for _ in range(batch_size):
        current0, current1 = recipes[index0], recipes[index1]
        total = current0 + current1
        if total >= 10:
          recipes[num] = 1
          recipes[num + 1] = total - 10
          num += 2
        else:
          recipes[num] = total
          num += 1
        index0 += 1 + current0
        if index0 >= num:
          index0 %= num
        index1 += 1 + current1
        if index1 >= num:
          index1 %= num

      # Find pattern in new batch results using Boyer-Moore-Horspool.
      seq = recipes[prev_num:num]
      subseq = pattern
      m = len(subseq)
      n = len(seq)
      i = 0
      while i + m <= n:
        j = m - 1
        e = e_last = seq[i + j]
        while True:
          if e != subseq[j]:
            i += skip_table[e_last]
            break
          if j == 0:
            return prev_num + i
          j -= 1
          e = seq[i + j]

  return func(pattern)


check_eq(process2('51589'), 9)
check_eq(process2('01245'), 5)
check_eq(process2('92510'), 18)
check_eq(process2('59414'), 2018)
puzzle.verify(2, process2)  # ~170 ms.

# %% [markdown]
# <a name="day15"></a>
# ## Day 15: Combat simulation

# %% [markdown]
# - Part 1: What is the outcome of the combat described in your puzzle input?
#
# - Part 2: After increasing the Elves' attack power until it is just barely enough for them to win without any Elves dying, what is the outcome of the combat described in your puzzle input?

# %%
puzzle = advent.puzzle(day=15)

# %%
s1 = """
#######
#.G...#
#...EG#
#.#.#G#
#..G#E#
#.....#
#######
"""

s10 = """
#######
#G..#E#
#E#E.E#
#G.##.#
#...#E#
#...E.#
#######
"""

s11 = """
#######
#E..EG#
#.#G.E#
#E.##E#
#G..#.#
#..E#.#
#######
"""

s12 = """
#######
#E.G#.#
#.#G..#
#G.#.G#
#G..#.#
#...E.#
#######
"""

s13 = """
#######
#.E...#
#.#..G#
#.###.#
#E#G#G#
#...#G#
#######
"""

s14 = """
#########
#G......#
#.E.#...#
#..##..G#
#...##..#
#...#...#
#.G...G.#
#.....G.#
#########
"""


# %%
def simulate_combat(s, verbose=False,
                    elf_attack_power=3, fail_if_elf_dies=False):

  @dataclasses.dataclass
  class Unit:
    yx: tuple[int, int]
    ch: str  # 'E' (Elf) or 'G' (Goblin).
    hit_points: int = 200

  show = hh.show if verbose else lambda *a, **k: None
  grid = hh.grid_from_string(s)
  units = [Unit(yx, ch) for ch in 'GE' for yx in zip(*np.nonzero(grid == ch))]

  def get_opponents(unit):
    return (u for u in units if u.ch != unit.ch)

  def manhattan(yx1, yx2):
    return abs(yx1[0] - yx2[0]) + abs(yx1[1] - yx2[1])

  def get_adjacent_opponents(unit):
    return [u for u in get_opponents(unit) if manhattan(u.yx, unit.yx) == 1]

  def empty_adjacent_yxs(yx):
    y, x = yx
    for yx2 in ((y - 1, x), (y, x - 1), (y, x + 1), (y + 1, x)):
      # Bounds checking is unnecessary because maze is surrounded by walls.
      if grid[yx2] == '.':
        yield yx2

  def adjacent_towards_opponent(inrange):
    # BFS from unit until yx in inrange; record all others at same distance.
    next_queue = [unit.yx]
    visited = set()
    nearests = set()
    for nearest_distance in itertools.count():
      if nearests or not next_queue:
        break
      queue, next_queue = next_queue, []
      for yx in queue:
        for yx2 in empty_adjacent_yxs(yx):
          if yx2 not in visited:
            visited.add(yx2)
            next_queue.append(yx2)
            if yx2 in inrange:
              nearests.add(yx2)
    if not nearests:
      return None  # No path to opponent.
    nearest = min(nearests)

    # BFS from nearest to unit; find unit neighbor with shortest distance.
    next_queue = [nearest]
    distances = {nearest: 0}
    for distance in range(nearest_distance - 1):
      queue, next_queue = next_queue, []
      for yx in queue:
        for yx2 in empty_adjacent_yxs(yx):
          if yx2 not in distances:
            if distance + manhattan(yx2, unit.yx) <= nearest_distance:  # A*
              distances[yx2] = distance + 1
              next_queue.append(yx2)
    return min(empty_adjacent_yxs(unit.yx),
               key=lambda yx: (distances.get(yx, math.inf), yx))

  for round in itertools.count():
    hit_points = [u.hit_points for u in sorted(units, key=lambda u: u.yx)]
    show('\nBegin round', round, hit_points)
    # show('' + hh.string_from_grid(grid))
    incomplete_round = False
    for unit in sorted(units, key=lambda unit: unit.yx):
      if unit not in units:  # If already deleted, skip it.
        continue
      if len(set(u.ch for u in units)) == 1:
        incomplete_round = True  # Unit cannot attack, so end of combat.
        break

      # Find nearest attacker-adjacent position and try to move towards it.
      adjacent_opponents = get_adjacent_opponents(unit)
      if not adjacent_opponents:
        inrange = {yx for u in get_opponents(unit)
                   for yx in empty_adjacent_yxs(u.yx)}
        if not inrange:  # No accessible opponent.
          continue
        best_adjacent = adjacent_towards_opponent(inrange)
        if best_adjacent is None:
          continue
        check_eq(grid[unit.yx], unit.ch)
        check_eq(grid[best_adjacent], '.')
        check_eq(manhattan(best_adjacent, unit.yx), 1)
        grid[unit.yx] = '.'
        grid[best_adjacent] = unit.ch
        unit.yx = best_adjacent
        adjacent_opponents = get_adjacent_opponents(unit)

      # Find adjacent opponent with lowest hit points and attack it.
      if adjacent_opponents:
        opponent = min(adjacent_opponents, key=lambda u: (u.hit_points, u.yx))
        attack_power = elf_attack_power if unit.ch == 'E' else 3
        opponent.hit_points -= attack_power
        if opponent.hit_points <= 0:
          check_eq(grid[opponent.yx], opponent.ch)
          grid[opponent.yx] = '.'
          units.remove(opponent)
          if fail_if_elf_dies and opponent.ch == 'E':
            return None
    if incomplete_round:
      break

  num_rounds = round
  hit_points = [u.hit_points for u in sorted(units, key=lambda u: u.yx)]
  sum_points = sum(hit_points)
  show(num_rounds, sum_points, hit_points)
  # show('' + hh.string_from_grid(grid))
  return num_rounds * sum_points

process1 = simulate_combat

check_eq(process1(s1), 27730)
check_eq(process1(s10), 36334)
check_eq(process1(s11), 39514)
check_eq(process1(s12), 27755)
check_eq(process1(s13), 28944)
check_eq(process1(s14), 18740)

puzzle.verify(1, process1)  # ~800 ms.


# %%
def simulate_combat(s, visualize=False, elf_attack_power=3,
                    fail_if_elf_dies=False):  # Using numba; optimized.

  @dataclasses.dataclass
  class Unit:
    yx: tuple[int, int]
    ch: str  # 'E' (Elf) or 'G' (Goblin).
    hit_points: int = 200

  grid = hh.grid_from_string(s)
  units = [Unit(yx, ch) for ch in 'GE' for yx in zip(*np.nonzero(grid == ch))]

  def get_opponents(unit):
    return (u for u in units if u.ch != unit.ch)

  def manhattan(yx1, yx2):
    return abs(yx1[0] - yx2[0]) + abs(yx1[1] - yx2[1])

  def get_adjacent_opponents(unit):
    return [u for u in get_opponents(unit) if manhattan(u.yx, unit.yx) == 1]

  def empty_adjacent_yxs(yx):
    y, x = yx
    for yx2 in ((y - 1, x), (y, x - 1), (y, x + 1), (y + 1, x)):
      # Bounds checking is unnecessary because maze is surrounded by walls.
      if grid[yx2] == '.':
        yield yx2

  # Passing a `set` into numba is ok but deprecated; frozenset is not supported.
  import warnings
  warnings_context = warnings.catch_warnings()
  warnings_context.__enter__()
  warnings.simplefilter(
      'ignore', category=numba.core.errors.NumbaPendingDeprecationWarning)

  @numba_njit(cache=True)
  def adjacent_towards_opponent(grid, unit_yx, inrange):
    # BFS from unit until yx in inrange; record all others at same distance.
    next_queue = [unit_yx]
    visited = set()
    nearests = set()
    for nearest_distance in range(10**8):
      if nearests or not next_queue:
        break
      queue, next_queue = next_queue, []
      for yx in queue:
        y, x = yx
        for yx2 in ((y - 1, x), (y, x - 1), (y, x + 1), (y + 1, x)):
          if grid[yx2] == '.' and yx2 not in visited:
            visited.add(yx2)
            next_queue.append(yx2)
            if yx2 in inrange:
              nearests.add(yx2)
    if not nearests:
      return None  # No path to opponent.
    nearest = min(nearests)

    # BFS from nearest to unit; find unit neighbor with shortest distance.
    next_queue = [nearest]
    distances = {nearest: 0}
    for distance in range(nearest_distance - 1):
      queue, next_queue = next_queue, []
      for yx in queue:
        y, x = yx
        for yx2 in ((y - 1, x), (y, x - 1), (y, x + 1), (y + 1, x)):
          if grid[yx2] == '.' and yx2 not in distances:
            remaining = abs(unit_yx[0] - yx2[0]) + abs(unit_yx[1] - yx2[1])
            if distance + remaining <= nearest_distance:  # A*.
              distances[yx2] = distance + 1
              next_queue.append(yx2)

    y, x = unit_yx
    best = 10**8, (-1, -1)
    for yx2 in ((y - 1, x), (y, x - 1), (y, x + 1), (y + 1, x)):
      if grid[yx2] == '.' and yx2 in distances and (distances[yx2], yx2) < best:
        best = distances[yx2], yx2
    return best[1]

  images = []
  incomplete_round = False
  for round in itertools.count():
    if visualize:
      cmap = {'.': (250,) * 3, '#': (0, 0, 0),
              'E': (255, 0, 0), 'G': (0, 190, 0)}
      image = np.array([cmap[e] for e in grid.ravel()], dtype=np.uint8)
      image = image.reshape(*grid.shape, 3)
      image = image.repeat(5, axis=0).repeat(5, axis=1)
      images.append(image)
    if incomplete_round:
      break
    hit_points = [u.hit_points for u in sorted(units, key=lambda u: u.yx)]
    incomplete_round = False
    for unit in sorted(units, key=lambda unit: unit.yx):
      if unit not in units:  # If already deleted, skip it.
        continue
      if len(set(u.ch for u in units)) == 1:
        incomplete_round = True  # Unit cannot attack, so end of combat.
        break

      # Find nearest attacker-adjacent position and try to move towards it.
      adjacent_opponents = get_adjacent_opponents(unit)
      if not adjacent_opponents:
        inrange = {yx for u in get_opponents(unit)
                   for yx in empty_adjacent_yxs(u.yx)}
        if not inrange:  # No accessible opponent.
          continue
        best_adjacent = adjacent_towards_opponent(grid, unit.yx, inrange)
        if best_adjacent is None:
          continue
        # check_eq(grid[unit.yx], unit.ch)
        # check_eq(grid[best_adjacent], '.')
        # check_eq(manhattan(best_adjacent, unit.yx), 1)
        grid[unit.yx] = '.'
        grid[best_adjacent] = unit.ch
        unit.yx = best_adjacent
        adjacent_opponents = get_adjacent_opponents(unit)

      # Find adjacent opponent with lowest hit points and attack it.
      if adjacent_opponents:
        opponent = min(adjacent_opponents, key=lambda u: (u.hit_points, u.yx))
        attack_power = elf_attack_power if unit.ch == 'E' else 3
        opponent.hit_points -= attack_power
        if opponent.hit_points <= 0:
          # check_eq(grid[opponent.yx], opponent.ch)
          grid[opponent.yx] = '.'
          units.remove(opponent)
          if fail_if_elf_dies and opponent.ch == 'E':
            return None

  warnings_context.__exit__()
  if visualize:
    images = [images[0]] * 20 + images + [images[-1]] * 30
    media.show_video(images, codec='gif', fps=10)
    return None
  num_rounds = round - 1
  hit_points = [u.hit_points for u in sorted(units, key=lambda u: u.yx)]
  sum_points = sum(hit_points)
  return num_rounds * sum_points

process1 = simulate_combat

check_eq(process1(s1), 27730)  # ~1800 ms for numba jit compilation!
check_eq(process1(s10), 36334)  # ~29 ms.
check_eq(process1(s11), 39514)
check_eq(process1(s12), 27755)
check_eq(process1(s13), 28944)
check_eq(process1(s14), 18740)
puzzle.verify(1, process1)  # ~210 ms.

# %%
process1(puzzle.input, visualize=True)  # ~600 ms.


# %%
def process2(s):  # Brute-force search of increasing attack power.
  for elf_attack_power in itertools.count(3):
    result = simulate_combat(
        s, elf_attack_power=elf_attack_power, fail_if_elf_dies=True)
    if result:
      print(f'Found solution at elf_attack_power={elf_attack_power}')
      return result

check_eq(process2(s1), 4988)
check_eq(process2(s11), 31284)
check_eq(process2(s12), 3478)
check_eq(process2(s13), 6474)
check_eq(process2(s14), 1140)

# puzzle.verify(2, process2)  # ~1600 ms.

# %%
def process2(s, visualize=False):  # Faster bisection search.
  current = 10
  low = high = None
  results = {}
  while True:
    results[current] = result = simulate_combat(
        s, elf_attack_power=current, fail_if_elf_dies=True)
    if result:  # We should try to decrease current.
      high = current
      if low is None:
        current = current // 2
      elif low < current - 1:
        current = (low + high) // 2
      else:
        break
    else:  # We need to increase current.
      low = current
      if high is None:
        current = current * 2
      elif current < high - 1:
        current = (low + high) // 2
      else:
        current = current + 1
        break
  print(f'Found solution at elf_attack_power={current}')
  if visualize:
    simulate_combat(
        s, elf_attack_power=current, fail_if_elf_dies=True, visualize=True)
  return results[current]

check_eq(process2(s1), 4988)
check_eq(process2(s11), 31284)
check_eq(process2(s12), 3478)
check_eq(process2(s13), 6474)
check_eq(process2(s14), 1140)

puzzle.verify(2, process2)  # ~650 ms. (elf_attack_power = 19; num_rounds = 40)

# %%
_ = process2(puzzle.input, visualize=True)  # ~1300 ms.

# %% [markdown]
# <a name="day16"></a>
# ## Day 16: Inferring opcodes

# %% [markdown]
# - Part 1: How many samples in your puzzle input behave like three or more opcodes?
#
# - Part 2: What value is contained in register 0 after executing the test program?

# %%
puzzle = advent.puzzle(day=16)

# %%
s1 = """
Before: [3, 2, 1, 1]
9 2 1 2
After:  [3, 2, 2, 1]
"""


# %%
def process1(s, part2=False):
  s1, s2 = s.split('\n\n\n')
  machine = Machine(num_registers=4)
  num_operations = len(machine.operations)
  candidates = {op: set(range(num_operations)) for op in machine.operations}
  num_compatible_with_3_or_more = 0
  examples = s1.strip('\n').split('\n\n')
  for example in examples:
    lines = example.split('\n')
    check_eq(len(lines), 3)
    before = list(map(int, lines[0][9:-1].split(',')))
    codes = list(map(int, lines[1].split()))
    after = list(map(int, lines[2][9:-1].split(',')))
    check_eq(len(codes), 4)
    opcode, *operands = codes
    assert 0 <= opcode < num_operations
    num_compatible = 0
    for operation, function in machine.operations.items():
      registers = before.copy()
      function(registers, operands)
      if registers == after:
        num_compatible += 1
      else:
        candidates[operation] -= {opcode}
    num_compatible_with_3_or_more += num_compatible >= 3

  if not part2:
    return num_compatible_with_3_or_more

  operation_from_opcode = {}
  while candidates:
    operation = next(op for op, set_ in candidates.items() if len(set_) == 1)
    opcode, = candidates.pop(operation)
    operation_from_opcode[opcode] = operation
    for set_ in candidates.values():
      set_ -= {opcode}
  print(operation_from_opcode)

  machine = Machine(num_registers=4)
  for line in s2.strip('\n').split('\n'):
    codes = list(map(int, line.split()))
    check_eq(len(codes), 4)
    opcode, *operands = codes
    operation = operation_from_opcode[opcode]
    machine.operations[operation](machine.registers, operands)

  return machine.registers[0]


check_eq(process1(s1 + '\n\n\n'), 1)
puzzle.verify(1, process1)  # ~20 ms.

process2 = functools.partial(process1, part2=True)
puzzle.verify(2, process2)  # ~22 ms.

# %% [markdown]
# <a name="day17"></a>
# ## Day 17: Water falling over reservoirs

# %% [markdown]
# - Part 1: How many tiles can the water reach within the range of y values in your scan?
#
# - Part 2: How many water tiles are left after the water spring stops producing water and all remaining water not at rest has drained?

# %%
puzzle = advent.puzzle(day=17)

# %%
s1 = """
x=495, y=2..7
y=7, x=495..501
x=501, y=3..7
x=498, y=2..4
x=506, y=1..2
x=498, y=10..13
x=504, y=10..13
y=13, x=498..504
"""


# %%
def process1(s, part2=False, visualize=False):
  grid = {}

  for line in s.strip('\n').split('\n'):
    pattern = r'[xy]=(\d+), [xy]=(\d+)\.\.(\d+)'
    x, y0, y1 = map(int, re.fullmatch(pattern, line).groups())
    for y in range(y0, y1 + 1):
      yx = (y, x) if line[0] == 'x' else (x, y)
      grid[yx[0], yx[1]] = '#'
  ymin, ymax = min(y for y, x in grid), max(y for y, x in grid)

  def encode(yx): return -yx[0], yx[1]
  def decode(yx): return -yx[0], yx[1]
  yx = 0, 500
  sources = [encode(yx)]  # heap
  grid[yx] = '|'

  while sources:
    yx = decode(sources[0])
    check_eq(grid[yx], '|')
    if yx[0] == ymax:
      heapq.heappop(sources)
      continue
    y1x = yx[0] + 1, yx[1]
    ch = grid.get(y1x, '.')
    if ch == '|':
      heapq.heappop(sources)
      continue
    if ch == '.':
      grid[y1x] = '|'
      heapq.heappush(sources, encode(y1x))
      continue
    assert ch in '~#'
    heapq.heappop(sources)
    bounded = []
    for dx in (-1, 1):
      yx2 = yx
      while True:
        yx2 = yx2[0], yx2[1] + dx
        ch2 = grid.get(yx2, '.')
        assert ch2 != '~'
        if ch2 == '#':
          bounded.append(yx2[1])
          break
        grid[yx2] = '|'
        y1x2 = yx2[0] + 1, yx2[1]
        if ch2 == '.' and y1x2 not in grid:
          heapq.heappush(sources, encode(yx2))
        if grid.get(y1x2, '.') not in '~#':
          break
    if len(bounded) == 2:
      for x in range(bounded[0] + 1, bounded[1]):
        grid[yx[0], x] = '~'

  if visualize:
    cmap = {'.': (250,) * 3, '#': (0, 0, 0),
            '~': (0, 0, 255), '|': (150, 150, 255)}
    image = hh.image_from_yx_map(grid, '.', cmap, pad=1).transpose(1, 0, 2)
    media.show_image(image, title='(transposed)', border=True)
    return None

  desired = '~' if part2 else '~|'
  return sum(1 for yx, ch in grid.items()
             if ymin <= yx[0] <= ymax and ch in desired)


check_eq(process1(s1), 57)
puzzle.verify(1, process1)  # ~110 ms.
process1(puzzle.input, visualize=True)

process2 = functools.partial(process1, part2=True)
check_eq(process2(s1), 29)
puzzle.verify(2, process2)  # ~110 ms.

# %% [markdown]
# <a name="day18"></a>
# ## Day 18: Cellular automaton

# %% [markdown]
# - Part 1: What will the total resource value of the lumber collection area be after 10 minutes?
#
# - Part 2: What will the total resource value of the lumber collection area be after 1000000000 minutes?

# %%
puzzle = advent.puzzle(day=18)

# %%
s1 = """
.#.#...|#.
.....#|##|
.|..|...#.
..|#.....#
#.#|||#|#|
...#.||...
.|....|...
||...#|.#|
|.||||..|.
...#.|..|.
"""


# %%
def process1(s, num_minutes=10, part2=False, visualize=False):
  grid = hh.grid_from_string(s)

  def evolve_grid():
    grid2 = np.pad(grid, 1, constant_values='.')
    yxs = set(itertools.product((-1, 0, 1), repeat=2)) - {(0, 0)}
    neighbors = np.array([np.roll(grid2, yx, (0, 1))[1:-1, 1:-1] for yx in yxs])
    num_adjacent_trees = (neighbors == '|').sum(axis=0)
    num_adjacent_lumberyards = (neighbors == '#').sum(axis=0)
    old_grid = grid.copy()
    grid[(old_grid == '.') & (num_adjacent_trees >= 3)] = '|'
    grid[(old_grid == '|') & (num_adjacent_lumberyards >= 3)] = '#'
    grid[(old_grid == '#') & ((num_adjacent_trees < 1) |
                              (num_adjacent_lumberyards < 1))] = '.'

  def resource_value():
    num_trees = np.count_nonzero(grid == '|')
    num_lumberyards = np.count_nonzero(grid == '#')
    return num_trees * num_lumberyards

  if not part2:
    for _ in range(num_minutes):
      evolve_grid()
    return resource_value()

  # Detect a repeating cycle to speed up the evolution.
  images = []
  configs = {}  # hashed_grid -> remaining minute it first appeared.
  remaining_minutes = 1_000_000_000
  period = None
  for minute in itertools.count():
    if visualize:
      cmap = {'.': (200, 0, 0), '|': (0, 200, 0), '#': (0, 0, 200)}
      image = np.array([cmap[e] for e in grid.ravel()], dtype=np.uint8)
      image = image.reshape(*grid.shape, 3).repeat(3, axis=0).repeat(3, axis=1)
      images.append(image)
    config = grid.tobytes()  # Hashable; ''.join(grid.flat) is slower.
    if config in configs and not period:
      period = configs[config] - remaining_minutes
      remaining_minutes = remaining_minutes % period
      if 0:
        print(f'At minute {minute}, found cycle with period {period}.')
    if not remaining_minutes:
      break
    configs[config] = remaining_minutes
    evolve_grid()
    remaining_minutes -= 1

  if visualize:
    videos = {
        'Start': [images[0]] * 20 + images[:50] + [images[49]] * 10,
        f'Cycling period={period}': images[-period:],
    }
    media.show_videos(videos, codec='gif', fps=10)
    return None
  return resource_value()


check_eq(process1(s1), 1147)
# process1(puzzle.input, num_minutes=400)
puzzle.verify(1, process1)  # ~10 ms.

process2 = functools.partial(process1, part2=True)
puzzle.verify(2, process2)  # ~440 ms.

# %%
_ = process2(puzzle.input, visualize=True)

# %% [markdown]
# <a name="day19"></a>
# ## Day 19: CPU with instruction pointer

# %% [markdown]
# - Part 1: What value is left in register 0 when the background process halts?
#
# - Part 2: What value is left in register 0 when register 0 is started with the value 1?

# %%
puzzle = advent.puzzle(day=19)

# %%
s1 = """
#ip 0
seti 5 0 1
seti 6 0 2
addi 0 1 0
addr 1 2 3
setr 1 0 0
seti 8 0 4
seti 9 0 5
"""

# %%
# Part 1:
# 0 (0, 0, 0, 16, 0, 0)
# 1 (0, 0, 0, 17, 0, 2)
# 2 (0, 0, 0, 18, 0, 4)
# 3 (0, 0, 0, 19, 0, 76)
# 4 (0, 0, 0, 20, 0, 836)
# 5 (0, 6, 0, 21, 0, 836)
# 6 (0, 132, 0, 22, 0, 836)
# 7 (0, 145, 0, 23, 0, 836)
# 8 (0, 145, 0, 24, 0, 981)
# 9 (0, 145, 0, 25, 0, 981)
# 10 (0, 145, 0, 0, 0, 981)  # func_ip1(a=0, f=981)
# ...
# 11 (0, 145, 0, 1, 1, 981)
# 12 (0, 145, 1, 2, 1, 981)
# 13 (0, 1, 1, 3, 1, 981)
# 14 (0, 0, 1, 4, 1, 981)
# 15 (0, 0, 1, 5, 1, 981)
# 16 (0, 0, 1, 7, 1, 981)
# 17 (0, 0, 2, 8, 1, 981)
# 18 (0, 0, 2, 9, 1, 981)
# 19 (0, 0, 2, 10, 1, 981)
# 20 (0, 0, 2, 2, 1, 981)
# ...
# 7702823 (1430, 1, 982, 256, 982, 981)

# Part 2:
# 18 (0, 10550400, 0, 0, 0, 10551381)  # func_ip1(a=0, f=10551381)

# Registers abcdef  (d=IP)
#  1 seti 1 0 4  e = 1
#  2 seti 1 7 2  c = 2
#  3 mulr 4 2 1  # b = e * c
#  4 eqrr 1 5 1
#  5 addr 1 3 3  if e * c == f goto 7
#  6 addi 3 1 3  goto 8
#  7 addr 4 0 0  a += e
#  8 addi 2 1 2  c += 1
#  9 gtrr 2 5 1
# 10 addr 3 1 3  if c > f goto 12
# 11 seti 2 6 3  goto 3
# 12 addi 4 1 4  e += 1
# 13 gtrr 4 5 1
# 14 addr 1 3 3  if e > f goto 16
# 15 seti 1 3 3  goto 2

# def func_ip1(a, f):
#   for e in range(1, f + 1):
#     for c in range(2, f + 1):
#       if e * c == f:
#         a += e

# def func_equivalent(a, f):
#   a += sum(e for e in range(1, f + 1) if f % e == 0)

# def func_equivalent2(a, f):
#   a += sum(factors(f))

# %%
def process1(s, part2=False, verbose=False):

  def factors(n):
    result = set()
    for i in range(1, int(n ** 0.5) + 1):
      div, mod = divmod(n, i)
      if mod == 0:
        result |= {i, div}
    return result

  machine = Machine()
  machine.read_instructions(s)
  optimize = len(machine.instructions) > 10
  if optimize:
    check_eq(machine.instructions[4].operation, 'eqrr')
    register_f = machine.instructions[4].operands[1]
    check_eq(machine.instructions[7].operation, 'addr')
    register_a = machine.instructions[7].operands[2]
  if part2:
    machine.registers[0] = 1
  history = []
  while 0 <= machine.ip < len(machine.instructions):
    if optimize and machine.ip == 1:
      # Unoptimized, part1 is slow (~8 s) and part2 is way too slow.
      f = machine.registers[register_f]
      machine.registers[register_a] += sum(factors(f))
      machine.ip = 16
    else:
      machine.run_instruction()
    history.append(tuple(machine.registers))
    if len(history) > 10_000_000:
      break

  if verbose:
    for i in range(min(200, len(history))):
      print(i, history[i])
    if len(history) > 200:
      for i in range(0, len(history), 100_000):
        print(i, history[i])
      for i in range(len(history) - 200, len(history)):
        print(i, history[i])

  return machine.registers[0]


check_eq(process1(s1), 6)
puzzle.verify(1, process1)  # ~0 ms.

process2 = functools.partial(process1, part2=True)
puzzle.verify(2, process2)  # ~1 ms.

# %% [markdown]
# <a name="day20"></a>
# ## Day 20: Regexp of doors in 2D map

# %% [markdown]
# - Part 1: What is the largest number of doors you would be required to pass through to reach a room?
#
# - Part 2: How many rooms have a shortest path from your current location that pass through at least 1000 doors?

# %%
puzzle = advent.puzzle(day=20)


# %%
# I created a general solution, which can work on inputs that define
# passageways with 2D cycles.  However, it appears that all instances of the
# puzzle inputs give rise to a simple tree of passageways?
# In the case of a tree, can one simply look for the longest expansion of the
# regular expression?  No, it appears that the regexp expansions may involve
# backtracking along edges of the tree.  My solution is likely overkill for
# this simpler case.

def process1(s, part2=False, visualize=False):
  s, = re.fullmatch(r'\^([SNEW(|)]+)\$', s.strip()).groups()

  def parse(s):
    l = []
    while s and s[0] not in '|)':
      i = next((i for i, ch in enumerate(s) if ch in '(|)'), len(s))
      if i:
        l.append(s[:i])
        s = s[i:]
      else:
        check_eq(s[0], '(')
        l2 = []
        while s[0] != ')':
          l3, s = parse(s[1:])
          l2.append(l3)
          assert s[0] in '|)', s[0]
        l.append(l2)
        s = s[1:]
    return l, s

  l, _ = parse(s)

  def traverse(l):  # Returns three sets: doors_s, doors_e, yxs.
    if not l:
      return set(), set(), set()
    half = len(l) // 2
    if half:
      doors_s1, doors_e1, yxs1 = traverse(l[:half])
      doors_s2, doors_e2, yxs2 = traverse(l[half:])
      return (doors_s1 | {(y + v, x + u) for y, x in yxs1 for v, u in doors_s2},
              doors_e1 | {(y + v, x + u) for y, x in yxs1 for v, u in doors_e2},
              {(y + v, x + u) for y, x in yxs1 for v, u in yxs2})
    elem, = l
    if isinstance(elem, str):
      doors_s, doors_e = set(), set()
      y, x = 0, 0
      for ch in elem:
        if ch == 'S':
          doors_s.add((y, x))
          y += 1
        elif ch == 'N':
          y -= 1
          doors_s.add((y, x))
        elif ch == 'E':
          doors_e.add((y, x))
          x += 1
        elif ch == 'W':
          x -= 1
          doors_e.add((y, x))
        else:
          raise AssertionError(ch)
      return doors_s, doors_e, {(y, x)}
    # isinstance(elem, list)
    # Return the three unions of the respective sets from all child nodes.
    return (set().union(*tup) for tup in zip(*map(traverse, elem)))

  doors_s, doors_e, _ = traverse(l)

  if 0 and visualize:
    def symbols_from_doors():
      map1 = {(y * 2 + 1, x * 2): '-' for y, x in doors_s}
      map2 = {(y * 2, x * 2 + 1): '|' for y, x in doors_e}
      return {**map1, **map2, (0, 0): 'X'}
    print(hh.string_from_grid(hh.grid_from_indices(
        symbols_from_doors(), background='.')))

  yx = 0, 0
  distances = {yx: 0}
  queue = collections.deque([yx])
  while queue:
    yx = queue.popleft()
    for yx2, present in [((yx[0] + 1, yx[1]), yx in doors_s),
                         ((yx[0] - 1, yx[1]), (yx[0] - 1, yx[1]) in doors_s),
                         ((yx[0], yx[1] + 1), yx in doors_e),
                         ((yx[0], yx[1] - 1), (yx[0], yx[1] - 1) in doors_e)]:
      if present and yx2 not in distances:
        distances[yx2] = distances[yx] + 1
        queue.append(yx2)

  if visualize:
    map1 = {}
    for y, x in doors_s:
      map1[y * 2, x * 2] = map1[y * 2 + 1, x * 2] = map1[y * 2 + 2, x * 2] = 1
    for y, x in doors_e:
      map1[y * 2, x * 2] = map1[y * 2, x * 2 + 1] = map1[y * 2, x * 2 + 2] = 1
    map1[0, 0] = 2
    if part2:
      for (y, x), distance in distances.items():
        if distance >= 1000:
          map1[y * 2, x * 2] = 3
    cmap = {0: (0,) * 3, 1: (250,) * 3, 2: (255, 0, 0), 3: (160, 140, 255)}
    image = hh.image_from_yx_map(map1, 0, cmap, pad=2)
    image = image.repeat(2, axis=0).repeat(2, axis=1)
    media.show_image(image, border=True, height=max(60, image.shape[0]))

  if not part2:
    return max(distances.values())

  return sum(1 for distance in distances.values() if distance >= 1000)


check_eq(process1('^WNE$'), 3)
check_eq(process1('^ENWWW(NEEE|SSE(EE|N))$'), 10)
check_eq(process1('^ENNWSWW(NEWS|)SSSEEN(WNSE|)EE(SWEN|)NNN$'), 18)
check_eq(process1('^ESSWWN(E|NNENN(EESS(WNSE|)SSS|WWWSSSSE(SW|NNNE)))$'), 23)
check_eq(process1('^WSSEESWWWNW(S|NENNEEEENN(ESSSSW(NWSW|SSEN)|WSWWN(E|WWS(E|SS))))$'), 31)
puzzle.verify(1, process1)  # ~840 ms.

# %%
process2 = functools.partial(process1, part2=True, visualize=False)
puzzle.verify(2, process2)  # ~840 ms.
_ = process2(puzzle.input, visualize=True)

# %%
if 0:  # Due to backtracking, one cannot simply look for longest expansion.
  def process1(s, part2=False, visualize=True):
    s, = re.fullmatch(r'\^([SNEW(|)]+)\$', s.strip()).groups()

    def max_regex_length(s):
      max_len = 0
      while s and s[0] not in '|)':
        i = next((i for i, ch in enumerate(s) if ch in '(|)'), len(s))
        if i:
          max_len += i
          s = s[i:]
        else:
          check_eq(s[0], '(')
          max_lens = []
          while s[0] != ')':
            len_child, s = max_regex_length(s[1:])
            max_lens.append(len_child)
            assert s[0] in '|)', s[0]
          max_len += max(max_lens)
          s = s[1:]
      return max_len, s

    max_len, _ = max_regex_length(s)
    return max_len

  check_eq(process1('^WNE$'), 3)
  check_eq(process1('^ENWWW(NEEE|SSE(EE|N))$'), 10)
  check_eq(process1('^ENNWSWW(NEWS|)SSSEEN(WNSE|)EE(SWEN|)NNN$'), 18)

# %% [markdown]
# <a name="day21"></a>
# ## Day 21: Smallest value causing halt

# %% [markdown]
# - Part 1: What is the lowest non-negative integer value for register 0 that causes the program to halt after executing the fewest instructions?
#
# - Part 2: What is the lowest non-negative integer value for register 0 that causes the program to halt after executing the most instructions?

# %%
puzzle = advent.puzzle(day=21)

# %%
def test():
  for i, line in enumerate(puzzle.input.splitlines()[1:]):
    print(f'{i:2} {line}')

if 0:
  test()

#  0 seti 123 0 4       # e = 123
#  1 bani 4 456 4       # e &= 456
#  2 eqri 4 72 4
#  3 addr 4 1 1         # if e == 72: goto 5
#  4 seti 0 0 1         # goto 1
#  5 seti 0 8 4         # e = 0
# BEGIN
#  6 bori 4 65536 3     # d = e | 65536
#  7 seti 16098955 8 4  # e = 16098955
#  8 bani 3 255 5       # f = d & 255
#  9 addr 4 5 4         # e += f
# 10 bani 4 16777215 4  # e &= 16777215
# 11 muli 4 65899 4     # e *= 65899
# 12 bani 4 16777215 4  # e &= 16777215
# 13 gtir 256 3 5
# 14 addr 5 1 1         # if d < 256: goto 16
# 15 addi 1 1 1         # goto 17
# 16 seti 27 3 1        # goto 28
# 17 seti 0 7 5         # f = 0
# 18 addi 5 1 2         # c = f + 1
# 19 muli 2 256 2       # c *= 256
# 20 gtrr 2 3 2
# 21 addr 2 1 1         # if c > d: goto 23
# 22 addi 1 1 1         # goto 24
# 23 seti 25 1 1        # goto 26
# 24 addi 5 1 5         # f += 1
# 25 seti 17 6 1        # goto 18
# 26 setr 5 4 3         # d = f
# 27 seti 7 5 1         # goto 8
# 28 eqrr 4 0 5
# 29 addr 5 1 1         # if e == a: goto 31 (halt)
# 30 seti 5 3 1         # goto 6

# def func(a):
#   d = 65536
#   e = 16098955
#   while True:
#     f = d & 255
#     e += f
#     e &= 16777215  # 0xFFFFFF (24-bit)
#     e *= 65899
#     e &= 16777215
#     if d < 256:
#       if e == a:
#         break  # halt!
#       d = e | 65536
#       e = 16098955
#       continue
#     f = 0
#     while True:
#       c = f + 1
#       c *= 256
#       if c > d:
#         d = f
#         break
#       f += 1

# def func(a):
#   d = 65536
#   e = 16098955
#   while True:
#     f = d & 255
#     e += f
#     e &= 16777215  # 0xFFFFFF (24-bit)
#     e *= 65899
#     e &= 16777215
#     if d < 256:
#       if e == a:
#         break  # halt!
#       d = e | 65536
#       e = 16098955
#       continue
#     d //= 256

# %%
def test():

  def simulate(max_count=10_000_000, verbose=False):
    d = 65536
    e = 16098955
    for _ in range(max_count):
      e += d & 255
      e &= 16777215  # 0xFFFFFF (24-bit)
      e *= 65899
      e &= 16777215
      if verbose:
        hh.show(a, d, e)
      if d < 256:
        yield e
        d = e | 65536
        e = 16098955
        continue
      d //= 256

  def count_for(a, max_count=10_000_000, verbose=False):
    for count, e in enumerate(simulate(verbose=verbose)):
      if count >= max_count:
        return None
      if e == a:
        return count + 1
    return None

  def sweep(max_a, max_count):
    for a in range(max_a):
      count = count_for(a, max_count=max_count)
      if count:
        hh.show(a, count)

  if 0:
    sweep(max_a=1_000_000, max_count=1_000)
    sweep(max_a=16_777_216, max_count=10)
    # a = 2014420, count = 5
    # a = 12063646, count = 8
    # a = 15823996, count = 2

  for a in [15823996]:
    count = count_for(a, verbose=True)
    hh.show(count)
  # a = 15823996, d = 65536, e = 14559001
  # a = 15823996, d = 256, e = 1732723
  # a = 15823996, d = 1, e = 15823996
  # count = 2
  # This makes sense --- it is the simplest solution -- I should have inferred
  # this earlier.

  def find_last_unique_element():
    results = {}
    for e in simulate(max_count=20_000_000):
      results[e] = 1
    print(len(results), list(results)[-5:])

  find_last_unique_element()
  # 11457 [9486379, 5590365, 182116, 12821901, 10199686]


# test()

# %%
def process1(s, part2=False):
  machine = Machine()
  machine.read_instructions(s)

  def gen_sequence():
    machine.registers = [0, 0, 0, 0, 0, 0]
    check_eq(machine.instructions[28].operation, 'eqrr')
    register_e = machine.instructions[28].operands[0]
    check_eq(machine.instructions[17].operation, 'seti')
    check_eq(machine.instructions[17].operands[0], 0)
    check_eq(machine.instructions[26].operation, 'setr')
    register_d = machine.instructions[26].operands[2]
    while True:
      assert 0 <= machine.ip < len(machine.instructions)
      if machine.ip == 28:  # Intercept generated number and disable halting.
        # 28 eqrr 4 0 5
        # 29 addr 5 1 1         # if e == a: goto 31 (halt)
        yield machine.registers[register_e]
        machine.ip = 30
        continue
      if machine.ip == 17:  # Speed up the "d //= 256" computation.
        # 17 seti 0 7 5         # f = 0
        # ...
        # 26 setr 5 4 3         # d = f
        machine.registers[register_d] //= 256
        machine.ip = 27
        continue
      machine.run_instruction()

  if not part2:
    return next(gen_sequence())

  results = {}
  for a in gen_sequence():
    if a in results:
      print(f'Found {len(results)} numbers before first repetition.')
      return list(results)[-1]  # Last number before first repetition.
    results[a] = 1


puzzle.verify(1, process1)  # ~0 ms.

process2 = functools.partial(process1, part2=True)
puzzle.verify(2, process2)  # ~630 ms.

# %% [markdown]
# <a name="day22"></a>
# ## Day 22: Shortest path with tools

# %% [markdown]
# - Part 1: What is the total risk level for the smallest rectangle that includes 0,0 and the target's coordinates?
#
# - Part 2: What is the fewest number of minutes you can take to reach the target?

# %%
puzzle = advent.puzzle(day=22)

# %%
s1 = """
depth: 510
target: 10,10
"""


# %% [markdown]
# The approach is to find shortest path over a 3D grid (y, x, item (0..2)), with some padding beyond the target location because the path may lie outside the the tight bounding box.
#
# I first try the useful graph library `networkx`, comparing the simple Dijkstra algorithm and the `A*` algorithm.  The `A*` algorithm is actually slower, which makes sense because the search domain is a rather tight box bounding the source and target locations.
#
# I obtain faster results using a manual implementation of the Dijkstra algorithm.  This is expected because the `networkx` data structure has overhead due to its many nested `dict` elements.

# %%
# https://www.reddit.com/r/adventofcode/comments/a8i1cy/comment/ecazvbe
def process1(s, part2=False, pad=60):  # Using networkx; slower.
  rocky, wet, narrow = 0, 1, 2
  del narrow  # unused
  torch, gear, neither = 0, 1, 2
  valid_items = {rocky: (torch, gear), wet: (gear, neither),
                 neither: (torch, neither)}

  def get_cave():
    lines = iter(line.strip() for line in s.strip('\n').split('\n'))
    depth = int(next(lines)[len('depth: '):])
    target = tuple(int(n) for n in next(lines)[len('target: '):].split(','))
    return depth, target

  def generate_grid(depth, shape):
    # (x, y) -> geologic index, erosion level, risk
    grid = {}
    for y, x in np.ndindex((shape[1], shape[0])):
      if (x, y) in [(0, 0), target]:
        geo = 0
      elif x == 0:
        geo = y * 48271
      elif y == 0:
        geo = x * 16807
      else:
        geo = grid[x-1, y][1] * grid[x, y-1][1]
      ero = (geo + depth) % 20183
      risk = ero % 3
      grid[x, y] = (geo, ero, risk)
    return grid

  def generate_graph(grid, shape):
    # Note: Using add_weighted_edges_from() just ends up calling add_edge().
    # Note: Using np.array for grid is actually slower.
    graph = nx.Graph()
    for y, x in np.ndindex((shape[1], shape[0])):
      items = valid_items[grid[x, y]]
      graph.add_edge((x, y, items[0]), (x, y, items[1]), weight=7)
    for y, x in np.ndindex((shape[1] - 1, shape[0] - 1)):
      items = set(valid_items[grid[x, y]])
      for dx, dy in ((0, 1), (1, 0)):
        new_x, new_y = x + dx, y + dy
        new_items = valid_items[grid[new_x, new_y]]
        for item in items.intersection(new_items):
          graph.add_edge((x, y, item), (new_x, new_y, item), weight=1)
    return graph

  def cost_lower_bound(xyi1, xyi2):
    (x1, y1, item1), (x2, y2, item2) = xyi1, xyi2
    return abs(x1 - x2) + abs(y1 - y2) + (7 if item1 != item2 else 0)

  depth, target = get_cave()
  if not part2:
    shape = (target[0] + 1, target[1] + 1)
    grid = generate_grid(depth, shape)
    return sum(v[2] for v in grid.values())

  import networkx as nx
  shape = (target[0] + pad, target[1] + pad)
  grid = {c: v[2] for c, v in (generate_grid(depth, shape)).items()}
  graph = generate_graph(grid, shape)
  use_astar = True
  if use_astar:
    return nx.astar_path_length(
        graph, (0, 0, torch), (*target, torch), heuristic=cost_lower_bound)
  return nx.dijkstra_path_length(graph, (0, 0, torch), (*target, torch))


check_eq(process1(s1), 114)
puzzle.verify(1, process1)  # ~15 ms.

if not importlib.util.find_spec('networkx'):
  print('Module networkx is unavailable.')
else:
  process2 = functools.partial(process1, part2=True)
  check_eq(process2(s1), 45)
  # puzzle.verify(2, process2)  # ~2000 ms using Dijkstra (~2.7 s using A*).


# %%
def process1(s, part2=False, pad=60, visualize=False):  # With numba.
  lines = s.strip('\n').split('\n')
  depth = int(re.fullmatch(r'depth: (\d+)', lines[0]).group(1))
  pattern = r'target: (\d+),(\d+)'
  target_yx = tuple(map(int, re.fullmatch(pattern, lines[1]).groups()))[::-1]

  def construct_grid(shape):
    erosion_level = np.empty(shape, dtype=np.int64)
    for y, x in np.ndindex(shape):
      if y == 0:
        geologic_index = x * 16807
      elif x == 0:
        geologic_index = y * 48271
      elif (y, x) == target_yx:
        geologic_index = 0
      else:
        geologic_index = erosion_level[y - 1, x] * erosion_level[y, x - 1]
      erosion_level[y, x] = (geologic_index + depth) % 20183
    return erosion_level % 3

  if not part2:
    shape = tuple(np.array(target_yx) + 1)
    return construct_grid(shape).sum()

  @numba_njit(cache=True)
  def dijkstra(grid, target_yx, visualize):
    # https://levelup.gitconnected.com/dijkstra-algorithm-in-python-8f0e75e3f16e
    TORCH = 1
    distances = {}
    visited = set([(0, 0, 0) for _ in range(0)])  # Typed empty set.
    parents = {(0, 0, 0): (0, 0, 0)}  # Dummy entry for numba typing.
    source_tyx = TORCH, 0, 0
    target_tyx = TORCH, *target_yx
    distances[source_tyx] = 0
    pq = [(0, source_tyx)]
    tool_change = np.array([[-1, 2, 1], [2, -1, 0], [1, 0, -1]])

    while pq:
      distance, node = heapq.heappop(pq)
      if node in visited:
        continue
      visited.add(node)
      if node == target_tyx:
        break

      def consider(node2, edge_cost):
        if ~(node2 in visited):  # (Workaround for numba bug "not in".)
          distance2 = distance + edge_cost
          if distance2 < distances.get(node2, 10**8):
            distances[node2] = distance2
            heapq.heappush(pq, (distance2, node2))
            if visualize:
              parents[node2] = node

      tool, y, x = node
      if y > 0 and grid[y - 1, x] != tool:
        consider((tool, y - 1, x), 1)
      if x > 0 and grid[y, x - 1] != tool:
        consider((tool, y, x - 1), 1)
      if y < grid.shape[0] - 1 and grid[y + 1, x] != tool:
        consider((tool, y + 1, x), 1)
      if x < grid.shape[1] - 1 and grid[y, x + 1] != tool:
        consider((tool, y, x + 1), 1)
      consider((tool_change[tool, grid[y, x]], y, x), 7)
    else:
      assert False

    path = [node]
    while node in parents:
      node = parents[node]
      path.append(node)
    return distance, path[::-1] if visualize else None

  shape = tuple(np.array(target_yx) + pad)
  grid = construct_grid(shape)
  distance, path = dijkstra(grid, target_yx, visualize)
  if visualize:
    cmap = {0: (150, 0, 0), 1: (0, 150, 0), 2: (0, 0, 150)}
    image = np.array([cmap[e] for e in grid.ravel()], dtype=np.uint8)
    image = image.reshape(*grid.shape, 3)
    image2 = image.copy()
    for node in path:
      image2[node[1:]] += 105  # Let the path be brighter.
    # x_max = (image.sum(2) > 0).max(axis=0).argmin() + 5
    image = image.transpose(1, 0, 2).repeat(2, axis=0).repeat(2, axis=1)
    image2 = image2.transpose(1, 0, 2).repeat(2, axis=0).repeat(2, axis=1)
    media.show_video([image, image2], codec='gif', fps=1)
  return distance


check_eq(process1(s1), 114)
puzzle.verify(1, process1)  # ~14 ms.

process2 = functools.partial(process1, part2=True)
check_eq(process2(s1), 45)  # 2.4 s for numba jit compilation!
puzzle.verify(2, process2)  # ~240 ms (~1050 ms without numba)
_ = process2(puzzle.input, visualize=True)  # ~1 s.

# %% [markdown]
# <a name="day23"></a>
# ## Day 23: Position in most octahedra

# %% [markdown]
# - Part 1: Find the nanobot with the largest signal radius. How many nanobots are in range of its signals?
#
# - Part 2: Find the coordinates that are in range of the largest number of nanobots. What is the shortest manhattan distance between any of those points and `0,0,0`?

# %%
puzzle = advent.puzzle(day=23)

# %%
s1 = """
pos=<0,0,0>, r=4
pos=<1,0,0>, r=1
pos=<4,0,0>, r=3
pos=<0,2,0>, r=1
pos=<0,5,0>, r=3
pos=<0,0,3>, r=1
pos=<1,1,1>, r=1
pos=<1,1,2>, r=1
pos=<1,3,1>, r=1
"""

# %%
s2 = """
pos=<10,12,12>, r=2
pos=<12,14,12>, r=2
pos=<16,12,12>, r=4
pos=<14,14,14>, r=6
pos=<50,50,50>, r=200
pos=<10,10,10>, r=5
"""


# %%
def process1(s, part2=False):
  positions, radii = [], []
  for line in s.strip('\n').split('\n'):
    pattern = r'pos=<([0-9-]+),([0-9-]+),([0-9-]+)>, r=(\d+)'
    x, y, z, r = map(int, re.fullmatch(pattern, line).groups())
    positions.append((x, y, z))
    radii.append(r)
  positions, radii = np.array(positions), np.array(radii)

  # With Manhattan distance, the in-range region in dimension D is a cocube
  # (e.g., an octahedron for D=3).  To efficiently represent intersections of
  # cocubes, we use a polytope (bounded convex polyhedron), defined as the
  # intersection of half-spaces (each hyperplane parallel to a cocube face).
  # The extent of the polytope is represented by an 2**D-dimensional vector `d`.
  # Each tuple coefficient d_i represents a hyperplane of a cocube face.
  # For D=3, each of the 8 half-spaces is a linear inequality on (x, y, z):
  #  h[0] * x + h[1] * y + h[2] * z <= d_i,
  # where face 0 has h = (-1, -1, -1),  face 1 has h = (-1, -1, 1), etc.

  hvalues = tuple(itertools.product((-1, 1), repeat=3))  # shape [8, 3]
  polytopes = [tuple(np.dot(h, position) + radius for h in hvalues)
               for position, radius in zip(positions, radii)]

  def num_in_range(position):
    return np.count_nonzero(abs(positions - position).sum(axis=1) <= radii)

  def point_in_polytope(position, polytope):
    return all(np.dot(h, position) <= d for h, d in zip(hvalues, polytope))

  def intersect(polytope1, polytope2):
    return tuple(np.minimum(polytope1, polytope2))

  def is_empty(polytope):
    # Note that half-space h[i] is in opposite direction of h[-(i + 1)].
    return (polytope[0] < -polytope[7] or polytope[1] < -polytope[6] or
            polytope[2] < -polytope[5] or polytope[3] < -polytope[4])

  def is_infinite(polytope):
    return min(polytope) == math.inf

  if not part2:
    if 1:  # Faster.
      i = radii.argmax()
      return np.count_nonzero(
          abs(positions - positions[i]).sum(axis=1) <= radii[i])
    # Sanity check on polytopes.
    polytope = polytopes[radii.argmax()]
    return sum(1 for p in positions if point_in_polytope(p, polytope))

  # Estimate an initial position with a high count of overlapping octahedra.
  good_position = tuple(max(positions, key=num_in_range))

  distance_threshold, prune_allowance = 10_000_000, 8  # len(polytopes2) = 529, max_count = 528; 0.4 s.

  while True:
    good_ds = tuple(np.dot(hvalue, good_position) for hvalue in hvalues)
    # hh.show(good_position, num_in_range(good_position))
    # good_position = (22000266, 38655032, 24842411), num_in_range(good_position) = 831

    # Modify a polytope to ignore any half-space whose linear boundary is
    # distant from good_position.
    def omit_far_halfspaces(polytope, good_ds):
      return tuple(d if abs(d - good_d) < distance_threshold else math.inf
                   for d, good_d in zip(polytope, good_ds))

    polytopes2 = [omit_far_halfspaces(p, good_ds) for p in polytopes]
    polytopes2 = [p for p in polytopes2 if not is_infinite(p)]

    # Heuristically order the polytopes.
    polytopes2 = sorted(polytopes2, key=lambda p: np.abs(p).min())
    polytopes2 = list(reversed(polytopes2))

    pieces = {}  # polytope -> count_of_original_octahedra
    for i, polytope in enumerate(polytopes2):
      new_pieces = {}
      for piece, count in pieces.items():
        piece2 = intersect(polytope, piece)
        if is_empty(piece2):
          continue
        new_pieces[piece2] = max(count + 1,
                                 pieces.get(piece2, 0),
                                 new_pieces.get(piece2, 0))
      pieces.update(new_pieces)
      pieces[polytope] = max(1, pieces.get(polytope, 0))
      if prune_allowance:
        pieces = {piece: count for piece, count in pieces.items()
                  if count > i - prune_allowance}

    max_count = max(pieces.values())
    max_list = [piece for piece, count in pieces.items() if count == max_count]
    best_polytope, = max_list
    if len(polytopes2) - max_count > prune_allowance:
      print('Warning: pruning too large to guarantee the optimal solution.')

    min_distance = math.inf
    # for c in set(itertools.product([-1, 0, 1], repeat=3)) - {0, 0, 0}:
    # for c in hvalues:
    for c in [-np.sign(good_position)]:
      c = np.array(c, dtype=np.float64)
      A_ub = np.array(hvalues)
      b_ub = np.nan_to_num(best_polytope, posinf=10**10)
      res = scipy.optimize.linprog(c, A_ub, b_ub, method='simplex')
      check_eq(res.success, True)
      # Handle the case of results with half-integer coordinates.
      upper = tuple(np.floor(np.mod(res.x, 1.0) + 1.499).astype(int))
      for delta in np.ndindex(upper):
        position = tuple((res.x + delta).astype(int))
        in_polytope = point_in_polytope(position, best_polytope)
        distance = np.abs(position).sum()
        if in_polytope:
          min_distance = min(min_distance, distance)
          best_position = position
    distance_from_good = abs(np.array(best_position) - good_position).sum()
    hh.show(best_position, min_distance, distance_from_good, num_in_range(best_position))
    # best_position = (26794906, 46607439, 21078785), min_distance = 94481130, distance_from_good = 16510673, num_in_range(best_position) = 977
    if distance_from_good <= distance_threshold:
      break
    good_position = best_position
    prune_allowance = 4

  return min_distance


check_eq(process1(s1), 7)
puzzle.verify(1, process1)  # ~30 ms.

process2 = functools.partial(process1, part2=True)
check_eq(process2(s2), 36)
# puzzle.verify(2, process2)  # ~2000 ms.

# %%
def process1(s, part2=False):
  # Divide-and-conquer using octree decomposition, inspired by
  # https://github.com/wimglenn/advent-of-code-wim/blob/master/aoc_wim/aoc2018/q23.py.
  pattern = r'pos=<([0-9-]+),([0-9-]+),([0-9-]+)>, r=(\d+)'
  data = np.array([list(map(int, re.fullmatch(pattern, line).groups()))
                   for line in s.strip('\n').split('\n')])
  xs, rs = data[:, :3], data[:, 3]
  i = rs.argmax()
  if not part2:
    return (abs(xs - xs[i]).sum(axis=1) <= rs[i]).sum()

  x0 = xs.min(axis=0)
  priority_queue = [(0, xs.ptp(axis=0).max(), abs(x0).sum(), *x0)]
  while priority_queue:
    n_out_of_range, s, d, *x = heapq.heappop(priority_queue)
    x = np.array(x)
    s //= 2
    if not s:
      # return d
      x0 = x
      break
    dx = np.array(list(itertools.product([0, 1], repeat=3))) * s
    for row in x + dx:  # Consider 8 octree child cells.
      # Maximize number in range = minimize number out of range.
      lo = np.maximum(row - xs, 0)
      hi = np.maximum(xs - row - s + 1, 0)  # (xs - row1);  row1 = row + s - 1
      n_out = ((lo + hi).sum(axis=1) > rs).sum()
      if n_out < len(rs):
        heapq.heappush(priority_queue, (n_out, s, abs(row).sum(), *row))
  else:
    assert False

  # Search around neighborhood of x0.
  r = 8
  for dx in itertools.product(range(-r, r + 1), repeat=3):
    x = x0 + dx
    n_out = (abs(xs - x).sum(axis=1) > rs).sum()
    n_out_of_range, d = min((n_out_of_range, d), (n_out, abs(x).sum()))

  return d

check_eq(process1(s1), 7)
puzzle.verify(1, process1)  # ~4 ms.

process2 = functools.partial(process1, part2=True)
check_eq(process2(s2), 36)
puzzle.verify(2, process2)  # ~285 ms.


# %%
def process1(s, part2=False):
  # Divide-and-conquer using octree decomposition, adapted from
  # https://github.com/wimglenn/advent-of-code-wim/blob/master/aoc_wim/aoc2018/q23.py.
  # Improved to be robust (not assuming cubes with power-of-two dimensions).
  pattern = r'pos=<([0-9-]+),([0-9-]+),([0-9-]+)>, r=(\d+)'
  data = np.array([list(map(int, re.fullmatch(pattern, line).groups()))
                   for line in s.strip('\n').split('\n')])
  xs, rs = data[:, :-1], data[:, -1]
  if not part2:
    i = rs.argmax()
    return (abs(xs - xs[i]).sum(axis=1) <= rs[i]).sum()

  xl, xh = xs.min(axis=0), xs.max(axis=0)
  pq = [(0, (xh - xl).max(), abs(xl).sum(), tuple(xl), tuple(xh))]
  while pq:
    n_out, s, d, xl, xh = heapq.heappop(pq)
    if s == 0:
      return d
    xm = (np.array(xl) + xh) // 2  # Partition into up to 8 octree child cells.
    for child_min_max in itertools.product(
        *(((l, m), (m + 1, h)) if m < h else ((l, h),)
          for l, m, h in zip(xl, xm, xh))):
      xl, xh = np.array(child_min_max).T
      n_out = ((np.maximum(xl - xs, 0) + np.maximum(xs - xh, 0)).sum(axis=1)
               > rs).sum()  # Maximize num in-range = minimize num out-of-range.
      heapq.heappush(
          pq, (n_out, (xh - xl).max(), abs(xl).sum(), tuple(xl), tuple(xh)))


check_eq(process1(s1), 7)
puzzle.verify(1, process1)  # ~4 ms.

process2 = functools.partial(process1, part2=True)
check_eq(process2(s2), 36)
puzzle.verify(2, process2)  # ~60 ms.

# %% [markdown]
# <a name="day24"></a>
# ## Day 24: Armies with groups of units

# %% [markdown]
# - Part 1: How many units would the winning army have?
#
# - Part 2: How many units does the immune system have left after getting the smallest boost it needs to win?

# %%
puzzle = advent.puzzle(day=24)

# %%
s1 = """
Immune System:
17 units each with 5390 hit points (weak to radiation, bludgeoning) with an attack that does 4507 fire damage at initiative 2
989 units each with 1274 hit points (immune to fire; weak to bludgeoning, slashing) with an attack that does 25 slashing damage at initiative 3

Infection:
801 units each with 4706 hit points (weak to radiation) with an attack that does 116 bludgeoning damage at initiative 1
4485 units each with 2961 hit points (immune to radiation; weak to fire, cold) with an attack that does 12 slashing damage at initiative 4
"""


# %%
def simulate_immune_fight(s, verbose=False, boost=0, immune_must_win=False):

  @dataclasses.dataclass
  class Group:
    army: Army
    id: int
    units: int
    hit_points: int  # (per_unit)
    attack_damage: int  # (per unit)
    attack_type: str
    initiative: int  # Higher initiative attacks first and wins ties.
    attributes: dict[str, set[str]]  # ['immune'] and ['weak']
    target: Group
    targeted: bool

    def __init__(self, army, id, line):
      self.army = army
      self.id = id
      pattern = (r'(\d+) units each with (\d+) hit points( \(.*\))? with'
                 r' an attack that does (\d+) (.*) damage at initiative (\d+)')
      (units, hit_points, attributes, attack_damage, attack_type,
       initiative) = re.fullmatch(pattern, line).groups()
      (self.units, self.hit_points, self.attack_damage, self.attack_type,
       self.initiative) = (int(units), int(hit_points), int(attack_damage),
                           attack_type, int(initiative))
      self.attributes = {'immune': set(), 'weak': set()}
      if attributes:
        for attribute in attributes[2:-1].split('; '):
          t, ss = attribute.split(' to ')
          for s in ss.split(', '):
            self.attributes[t].add(s)
      if boost and army.name == 'Immune System':
        self.attack_damage += boost

    def effective_power(self):
      return self.units * self.attack_damage

    def selection_order(self):
      return (-self.effective_power(), -self.initiative)

  @dataclasses.dataclass
  class Army:
    name: str
    groups: list[Group]

    def __init__(self, s):
      lines = s.split('\n')
      self.name = lines[0][:-1]
      self.groups = [Group(self, i + 1, line)
                     for i, line in enumerate(lines[1:])]

  armies = [Army(s_army) for s_army in s.strip('\n').split('\n\n')]

  def get_opponent(army):
    return next(army2 for army2 in armies if army2 != army)

  def compute_damage(group, target):
    assert group.army != target.army
    damage = group.effective_power()
    if group.attack_type in target.attributes['immune']:
      damage = 0
    if group.attack_type in target.attributes['weak']:
      damage *= 2
    return damage

  while True:
    if verbose:
      for army in armies:
        print(f'{army.name}:')
        for group in army.groups:
          print(f'Group {group.id} contains {group.units} units')
      print()

    if any(not army.groups for army in armies):
      break

    # Target selection.
    for army in armies:
      for group in army.groups:
        group.targeted = False
    for army in armies:
      opponent = get_opponent(army)
      for group in sorted(army.groups, key=lambda g: g.selection_order()):
        target = max(opponent.groups, key=lambda g: (
            not g.targeted, compute_damage(group, g),
            g.effective_power(), g.initiative))
        damage = compute_damage(group, target)
        if target.targeted or damage == 0:
          group.target = None
        else:
          group.target = target
          target.targeted = True
          if verbose and target:
            print(f'{army.name} group {group.id} would deal defending'
                  f' group {target.id} {damage} damage')
    if verbose:
      print()

    # Attacking.
    total_killed = 0
    for group in sorted((group for army in armies for group in army.groups),
                        key=lambda g: -g.initiative):
      if group.units == 0 or not group.target or group.target.units == 0:
        continue
      target = group.target
      damage = compute_damage(group, target)
      units_killed = min(damage // target.hit_points, target.units)
      if verbose:
        print(f'{group.army.name} group {group.id} attacks defending'
              f' group {target.id}, killing {units_killed} units')
      target.units -= units_killed
      total_killed += units_killed
    if verbose:
      print()
    if total_killed == 0:
      return None  # The fight is a draw.

    # Remove empty groups.
    for army in armies:
      army.groups = [group for group in army.groups if group.units > 0]

  army = next(army for army in armies if army.groups)
  if immune_must_win and army.name != 'Immune System':
    return None
  return sum(group.units for group in army.groups)


process1 = simulate_immune_fight

check_eq(process1(s1), 5216)
puzzle.verify(1, process1)  # ~100 ms.


# %%
def process2(s):

  def boost_result(boost):
    return simulate_immune_fight(s, boost=boost, immune_must_win=True)

  def binary_search(func, lower, upper):
    """Returns lowest x for which bool(func(x)) is True."""
    if 0:
      check_eq(func(lower), None)
      check_eq(func(upper) is not None, True)
    while lower + 1 < upper:
      mid = (lower + upper) // 2
      value = func(mid)
      if value:
        upper = mid
      else:
        lower = mid
    return upper

  boost = binary_search(boost_result, lower=0, upper=2_000)
  print(f'Found solution with boost={boost}')
  return boost_result(boost)


check_eq(simulate_immune_fight(s1, boost=1570), 51)
check_eq(process2(s1), 51)

puzzle.verify(2, process2)  # ~1300 ms.

# %% [markdown]
# <a name="day25"></a>
# ## Day 25: Clustering nearby 4D points

# %% [markdown]
# - Part 1: Given a set of 4D points, what is the number of clusters of points when edges connect points with Manhattan distance 3 or less?
#
# - Part 2: No second part on day 25.

# %%
puzzle = advent.puzzle(day=25)

# %%
s1 = """
 0,0,0,0
 3,0,0,0
 0,3,0,0
 0,0,3,0
 0,0,0,3
 0,0,0,6
 9,0,0,0
12,0,0,0
"""

s2 = """
-1,2,2,0
0,0,2,-2
0,0,0,-2
-1,2,0,0
-2,-2,-2,2
3,0,2,-1
-1,3,2,2
-1,0,-1,0
0,2,1,-2
3,0,0,0
"""

s3 = """
1,-1,0,1
2,0,-1,0
3,2,-1,0
0,0,3,1
0,0,-1,-1
2,3,-2,0
-2,2,0,0
2,-2,0,-1
1,-1,0,-1
3,2,0,2
"""

s4 = """
1,-1,-1,-2
-2,-2,0,1
0,2,1,3
-2,3,-2,1
0,2,3,-2
-1,-1,1,-2
0,-2,-1,0
-2,2,3,-1
1,2,2,0
-1,-2,0,-2
"""


# %%
def process1(s):  # Slower version.
  points = [tuple(map(int, l.split(','))) for l in s.strip('\n').split('\n')]

  union_find = hh.UnionFind[int]()
  num_edges = 0
  for i in range(len(points)):
    for j in range(i + 1, len(points)):
      p, q = points[i], points[j]
      if (abs(p[0] - q[0]) + abs(p[1] - q[1]) +
          abs(p[2] - q[2]) + abs(p[3] - q[3])) <= 3:
        num_edges += 1
        union_find.union(i, j)

  print(f'Graph has {len(points)} vertices and {num_edges} edges.')
  cluster_reps = {union_find.find(i) for i in range(len(points))}
  return len(cluster_reps)

check_eq(process1(s1), 2)
check_eq(process1(s2), 4)
check_eq(process1(s3), 3)
check_eq(process1(s4), 8)
puzzle.verify(1, process1)  # ~740 ms.


# %%
def process1(s):  # Faster version, using numpy to identify the graph edges.
  points = [[int(t) for t in l.split(',')] for l in s.strip('\n').split('\n')]
  points = np.array(points)
  union_find = hh.UnionFind[int]()
  edges = abs(points[None] - points.reshape(-1, 1, 4)).sum(axis=-1) <= 3
  for i, j in np.argwhere(edges):
    union_find.union(i, j)
  return len({union_find.find(i) for i in range(len(points))})

check_eq(process1(s1), 2)
check_eq(process1(s2), 4)
check_eq(process1(s3), 3)
check_eq(process1(s4), 8)
puzzle.verify(1, process1)  # ~93 ms.

# %%
puzzle.verify(2, lambda s: '')  # (No "Part 2" on last day.)
# (aocd does not allow a blank answer; the answer is not submitted)

# %% [markdown]
# <a name="timings"></a>
# ## Timings

# %%
advent.show_times()

# %%
if 0:  # Compute min execution times over several calls.
  advent.show_times(recompute=True, repeat=3)

# %%
if 0:  # Look for unwanted pollution of namespace.
  print(textwrap.fill(' '.join(var for var, value in globals().items() if (
      not var.startswith('_') and not repr(value).startswith(
          ('<module', '<class', 'typing.', 'functools.partial('))))))

# %%
if 0:  # Save puzzle inputs and answers to a compressed archive for downloading.
  # Create a new tar.gz file.
  hh.run(f"""tar -C ~/.config/aocd/'{PROFILE.replace("_", " ")}' -czf /content/data.tar.gz *.txt""")

# %%
hh.show_notebook_cell_top_times()

# %% [markdown]
# # End

# %% [markdown]
# <!-- For Emacs:
# Local Variables:
# fill-column: 80
# End:
# -->
