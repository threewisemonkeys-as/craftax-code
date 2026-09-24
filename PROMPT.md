The brief above says what this environment is. Nobody will explain its rules; you
discover them by acting and reading what happened.

There is one phase and one budget, and there is no way to give up — so spend what you
are given, and spend it on experiments rather than on hope. The budget belongs to the
run, and how much of it is yours is what `./act status` reports: usually all of it, and
on a run too long for one session a stint of it, after which this session is over and
another continues the same run from exactly where you stopped.

## Acting

`./act do` plays a batch of actions in order and **stops as soon as you die**,
discarding the rest — the state you planned against is gone. Pass `--plan "..."` with
each batch; it is recorded in the log beside those actions and becomes your briefing
history.

Every action counts against the budget. act refuses anything invalid and tells you why
— read the error, it is information about the environment.

**This world moves whether or not you do.** Light changes, things arrive, and your own
condition drains on its own. A batch of forty actions planned against a still picture
is forty actions spent in a world that has moved on. Batch hard when you are doing
something you understand — crossing open ground, repeating a thing that worked — and
one or two at a time when you are finding out.

## Memory

`logs.txt` holds the whole run: an entry per action with your plan, the action, what it
was worth, and the observation it produced. It is written for you, verbatim, and it is
the only complete record — your context is not, and on a run this long your context
will be compacted more than once. Its format is documented in its opening lines; read
them before anything else.

**Read the observations with code, not with your eyes.** `./python` has numpy and
Pillow. You may look at one directly and sometimes should — but a picture taken in by
eye comes with transcription errors, and those send you chasing phantoms. Comparing two
observations to find exactly what is different between them is the single most useful
operation you have; write it once, as a script, and keep it.

Never read hundreds of observations into your context. There are more of them than you
can hold, and each one costs you room you will want later. Pipe them through a script
and print the conclusion, not the picture.

Keep durable findings in `notes.md`: what each action does, what each thing you can see
is, what state you are in, what you have ruled out. Your context will compact; files
survive. Helper scripts are worth keeping too — a parser, a map you are building up, a
simulator you can re-run all beat re-deriving them.

If you are playing a stint, `notes.md` is also how you talk to the session that takes
over from you. It gets this workspace and nothing else — not your reasoning, not the
thing you were about to try next — so a finding you did not write down did not happen.
Write it as you go rather than at the end: you will not be told which action is your
last.

## Playing well

- Guess how something works, then spend **1-2 actions** checking the guess and compare
  what changed. Once a mechanic is nailed down, send **10-20 actions** in one batch
  rather than paying for a round trip each. `left*12` is twelve actions in one token.
- The reward is evidence and it is cheap to get wrong. A number arriving does not say
  which of the things you just did earned it, and some of what it reports is only your
  own condition changing. When a reward surprises you, the way to find out what caused
  it is to do the shortest thing that would produce it again. That is a diagnosis and
  not a plan: once you know what paid, the question is what it is *for*, and a cheap
  thing you can repeat a hundred times is the least likely answer. If the shortest
  thing that pays is also the first thing you tried, suspect you have found a property
  of the machinery you are being measured through rather than of the world you are
  being measured on. Spend the batch on the world.
- Break what you can see into distinct kinds of thing — position, appearance,
  behaviour, what happens when you act on them — and give each a role: you, the ground,
  what blocks you, what hurts you, what is worth something.
- A thing that did nothing may only have done nothing *there*. Before you write an
  action off as inert, try it again from somewhere else — facing something different,
  standing on something different, holding something different. Actions that need a
  precondition are exactly the ones a probe repeated from the opening state will never
  meet.
- A model that reproduces everything you have logged is not a model you have tested.
  You generated that log, and it may never once have entered the regime you are least
  sure about. Before you trust a rule, find the shortest sequence that would tell it
  apart from its most plausible rival, and run that one. Prefer the experiment that
  could embarrass you over the one that will confirm you.
- Dying is not the end of the run and not a wasted action. The observation it produced
  is the state you died in, and it is the only place that state is ever shown — read it
  before you move on. What you lost is what you were carrying and where you stood —
  which is most of what a life is worth, so it is a real loss and worth avoiding.
  Nothing you can play ends a life; if one ends, the world ended it.
- Shell and Python loops around `./act` are encouraged — branch on a parsed
  observation, repeat until something changes, search for a position. You are not
  limited to fixed action lists.
- Before the budget runs low, read back the questions in `notes.md` you never answered
  and spend what is left on the cheapest experiment that settles the most load-bearing
  one.

Play until `./act status` says you are done — the run over, or your stint spent. Do not
stop to ask questions — there is nobody to answer them.
