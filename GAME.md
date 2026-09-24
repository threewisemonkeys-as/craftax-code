You are playing a game you have never seen before. {observations}

Something in it is you. The rest of it is for you to work out — nothing here names
anything you will see, and nothing describes what any action does.

`./act status` prints where you are and what is playable now; `./act board` says where
the latest observation is. `./act do <action> <action> ...` plays actions in order —
time passes only when you act, so a `noop` is a real experiment. Any action may be
repeated: `left*12`.

There is one budget and it is shared. Every action counts against it, and nothing is
held back for later. {death} and the run carries on spending the same budget — what
dying cost you is the actions you had already spent. Nothing you can play ends a life;
only the world does that.

Acting is sometimes rewarded. A number comes back with an action when it was worth
something; part of what it measures is the change in your own condition, and part of it
is not. Nothing will tell you what earned it. What this run is judged on is the best
single life in it — one life, taken as far as it goes — so a total across lives is not
the thing to grow.

`./python` is a Python with numpy and Pillow, for reading the observations.

The format of `logs.txt`, what each line of a block means, which actions you have and
how large the budget is are documented in its opening lines. Read them before anything
else.
