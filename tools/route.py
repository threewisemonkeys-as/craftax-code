#!/usr/bin/env python3
"""The ceiling: a scripted player that walks the early tech tree.

`cc_humanrl` had a 199-action route that wins on all five of its games, and it was
the regression test for the whole port — if it stopped winning, the environment was
broken and every other number that run produced was worthless. Craftax has no win,
so the analogue is a route that reliably reaches **iron tools** from a fixed seed.
It is a wiring check first and a ceiling second: it says the world generates, the
map can be read, actions do what they say, and the achievement diffing sees it.

Two things make this different from a fixed action list.

**It is reactive, not recorded.** The world moves on its own — mobs wander into the
square you were walking to, and being blocked turns you rather than moving you — so
a list of 600 actions written against one playthrough would not survive its own
second run. This re-plans from the map after every step and verifies that each move
happened.

**It reads the true state**, which the agent never sees: the block grid, the player's
position, the inventory. That is the whole point of it being harness-side. Nothing
here is written anywhere a session can read, and `tools/` is not on the workspace's
path.

Not a claim about what is achievable — a competent human reaches 98% of the maximum
(notes/craftax-harness-plan.md, F10) and this stops at iron. It is the floor of
*competence*, sitting above the random floor and far below a person.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Action -> (drow, dcol), matching craftax's own DIRECTIONS table. `left` decreases
# the column, `up` decreases the row; a position is indexed (row, col) into the
# level's map.
DELTA = {"left": (0, -1), "right": (0, 1), "up": (-1, 0), "down": (1, 0)}


def blocks():
    """The block table, imported late so this module can be read without jax."""
    from craftax.craftax.constants import SOLID_BLOCKS, BlockType  # noqa: PLC0415

    return BlockType, set(SOLID_BLOCKS)


class Blocked(RuntimeError):
    """Nowhere to go for what was asked. The route gives up on that goal, not the run."""


class Router:
    """One scripted run. `game` is a CraftaxGame; every action goes through it."""

    def __init__(self, game, budget: int) -> None:
        self.game = game
        self.budget = budget
        self.used = 0
        self.played: list[str] = []
        self.BlockType, self.solid = blocks()

    # -- the world, as only the harness sees it ------------------------------ #

    @property
    def state(self):
        return self.game.state()

    def grid(self) -> np.ndarray:
        return np.asarray(self.state.map[int(self.state.player_level)])

    def at(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.state.player_position)

    def inv(self, name: str) -> int:
        return int(getattr(self.state.inventory, name))

    def need(self, name: str) -> int:
        return int(getattr(self.state, f"player_{name}"))

    def walkable(self, grid: np.ndarray) -> np.ndarray:
        """Where a land creature may stand: not solid, not water, not lava."""
        B = self.BlockType
        ok = np.ones(len(B), dtype=bool)
        for value in self.solid:
            ok[value] = False
        for kind in (B.WATER, B.LAVA, B.INVALID, B.OUT_OF_BOUNDS, B.DARKNESS):
            ok[kind.value] = False
        return ok[grid]

    def diggable(self, grid: np.ndarray) -> np.ndarray:
        """What the pickaxe in hand can clear, which then becomes walkable path.

        Iron on seed 0 is embedded in rock: five cells of it, and not one with a
        walkable neighbour. A router that only walks over open ground calls that
        world ironless, which is not a fact about the world — it is a fact about
        the router. Tunnelling is also how a person gets there.
        """
        B = self.BlockType
        held = self.inv("pickaxe")
        ok = np.zeros(len(B), dtype=bool)
        ok[B.TREE.value] = True  # and it pays wood on the way through
        if held >= 1:
            ok[B.STONE.value] = ok[B.COAL.value] = ok[B.PLANT.value] = True
        if held >= 2:
            ok[B.IRON.value] = True
        if held >= 3:
            ok[B.DIAMOND.value] = True
        return ok[grid]

    # -- acting -------------------------------------------------------------- #

    def act(self, token: str) -> bool:
        """Play one action. False once the budget is gone or the player is dead."""
        if self.used >= self.budget or not self.game.alive:
            return False
        self.game.play(token)
        self.used += 1
        self.played.append(token)
        return self.game.alive

    def neighbours(self) -> list[tuple[int, int]]:
        """The four cells around us that are actually on the map.

        The map does not wrap and numpy does not mind: `threats[(r, 48)]` on a
        48-wide level raises, which is how a route that had otherwise survived
        eight seeds died at the edge of the ninth.
        """
        rows, cols = self.grid().shape
        r, c = self.at()
        return [
            (r + dr, c + dc)
            for dr, dc in DELTA.values()
            if 0 <= r + dr < rows and 0 <= c + dc < cols
        ]

    def facing(self, cell: tuple[int, int]) -> str | None:
        """The action that turns toward an adjacent cell, if it is adjacent."""
        r, c = self.at()
        for token, (dr, dc) in DELTA.items():
            if (r + dr, c + dc) == cell:
                return token
        return None

    def face(self, cell: tuple[int, int]) -> bool:
        """Turn to look at an adjacent cell. Solid blocks and water turn you without
        moving you, which is what makes this one action rather than a dance."""
        token = self.facing(cell)
        if token is None:
            return False
        return self.act(token)

    # -- getting there ------------------------------------------------------- #

    def path_to(
        self, wanted: np.ndarray, adjacent: bool, dig: bool = True
    ) -> list[tuple[int, int]]:
        """Breadth-first to the nearest wanted cell, through rock if need be.

        With `adjacent`, the destination is a cell *next to* one that is wanted —
        which is where you have to stand to `do` anything to it. With `dig`, the
        search may also cross anything the pickaxe in hand can clear; those cells
        cost an extra swing each, which the search does not model and does not need
        to, since the alternative is usually no path at all.
        """
        grid = self.grid()
        free = self.walkable(grid)
        if dig:
            free = free | self.diggable(grid)
        start = self.at()
        rows, cols = grid.shape
        if adjacent:
            goal = np.zeros_like(wanted)
            for dr, dc in DELTA.values():
                goal |= np.roll(np.roll(wanted, -dr, axis=0), -dc, axis=1)
            goal &= free
        else:
            goal = wanted & free
        if not goal.any():
            raise Blocked("nothing of that kind is on this level")

        seen = {start}
        queue = deque([(start, [])])
        while queue:
            (r, c), trail = queue.popleft()
            if goal[r, c]:
                # Possibly empty: standing where you need to stand is arriving.
                # Requiring a non-empty path made "the table I just placed" read as
                # unreachable, which cost seed 0 every achievement after the second.
                return trail
            for dr, dc in DELTA.values():
                nxt = (r + dr, c + dc)
                if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                    continue
                if nxt in seen or not free[nxt]:
                    continue
                seen.add(nxt)
                queue.append((nxt, [*trail, nxt]))
        raise Blocked("no walkable path to anything of that kind")

    def mine(self, cell: tuple[int, int], swings: int = 8) -> bool:
        """Clear one adjacent block so it can be stood on."""
        if not self.face(cell):
            return False
        for _ in range(swings):
            if self.walkable(self.grid())[cell]:
                return True
            if not self.act("do"):
                return False
        return self.walkable(self.grid())[cell]

    def walk(
        self, wanted: np.ndarray, adjacent: bool = True, dig: bool = True,
        tries: int = 24,
    ) -> bool:
        """Walk to the nearest wanted cell, re-planning whenever a step fails.

        A step fails when something has moved into the square — mobs wander, and the
        map that was read a moment ago is already out of date. Re-planning rather
        than pushing on is the difference between a route and a recording.
        """
        for _ in range(tries):
            self.defend()
            try:
                trail = self.path_to(wanted, adjacent, dig)
            except Blocked:
                return False
            if not trail:
                return True
            for cell in trail:
                token = self.facing(cell)
                if token is None:
                    break
                if not self.walkable(self.grid())[cell] and not self.mine(cell):
                    break
                was = self.at()
                if not self.act(token):
                    return False
                if self.at() == was:
                    break  # blocked: re-plan from where we actually are
            else:
                return True
        return False

    def kind(self, *names: str) -> np.ndarray:
        grid = self.grid()
        want = np.zeros_like(grid, dtype=bool)
        for name in names:
            want |= grid == getattr(self.BlockType, name).value
        return want

    # -- goals ---------------------------------------------------------------- #

    def harvest(self, name: str, item: str, count: int, swings: int = 6) -> bool:
        """Stand next to a block of this kind, face it, and work at it."""
        while self.inv(item) < count:
            if not self.walk(self.kind(name)):
                return False
            grid = self.grid()
            target = next(
                (
                    (r + dr, c + dc)
                    for (r, c) in [self.at()]
                    for dr, dc in DELTA.values()
                    if 0 <= r + dr < grid.shape[0]
                    and 0 <= c + dc < grid.shape[1]
                    and grid[r + dr, c + dc] == getattr(self.BlockType, name).value
                ),
                None,
            )
            if target is None or not self.face(target):
                return False
            before = self.inv(item)
            for _ in range(swings):
                if not self.act("do"):
                    return False
                if self.inv(item) > before:
                    break
            else:
                return False
        return True

    def drink(self, upto: int = 9) -> bool:
        """Thirst is what kills an idle player first — drink hits zero around action
        200 and the health goes with it. Every long route has to do this."""
        if not self.walk(self.kind("WATER")):
            return False
        grid = self.grid()
        target = next(
            (cell for cell in self.neighbours()
             if grid[cell] == self.BlockType.WATER.value),
            None,
        )
        if target is None or not self.face(target):
            return False
        while self.need("drink") < upto:
            if not self.act("do"):
                return False
        return True

    def eat(self) -> bool:
        """Cows move, so this walks at where one was and works with what it finds."""
        state = self.state
        level = int(state.player_level)
        mask = np.asarray(state.passive_mobs.mask[level]).astype(bool)
        if not mask.any():
            return False
        grid = self.grid()
        want = np.zeros_like(grid, dtype=bool)
        for (r, c) in np.asarray(state.passive_mobs.position[level])[mask]:
            want[int(r), int(c)] = True
        if not self.walk(want):
            return False
        for cell in self.neighbours():
            if want[cell] and self.face(cell):
                before = self.need("food")
                for _ in range(4):
                    if not self.act("do"):
                        return False
                    if self.need("food") > before:
                        return True
        return False

    def mobs(self, kind: str) -> np.ndarray:
        """Where the mobs of one class are on this level, as a mask."""
        state = self.state
        level = int(state.player_level)
        group = getattr(state, kind)
        mask = np.asarray(group.mask[level]).astype(bool)
        want = np.zeros(self.grid().shape, dtype=bool)
        for (r, c) in np.asarray(group.position[level])[mask]:
            want[int(r), int(c)] = True
        return want

    def defend(self, swings: int = 4) -> bool:
        """Hit whatever is standing next to us.

        Night is what ends these runs: a zombie walks up while the route is busy
        pathing, and a router that only ever presses `do` at rock dies holding a
        sword. Cheap, because it acts only when something is actually adjacent.
        """
        threats = self.mobs("melee_mobs")
        for cell in self.neighbours():
            if not threats[cell]:
                continue
            if not self.face(cell):
                return False
            for _ in range(swings):
                if not self.act("do"):
                    return False
                if not self.mobs("melee_mobs")[cell]:
                    return True
            return True
        return False

    def upkeep(self) -> None:
        """Between goals, top up whatever is running out and clear what is close."""
        self.defend()
        if self.need("drink") <= 6:
            self.drink()
        if self.need("food") <= 6:
            self.eat()

    STATIONS = {"CRAFTING_TABLE": "place_table", "FURNACE": "place_furnace"}

    def beside_all(self, stations: tuple[str, ...]) -> np.ndarray:
        """Cells that touch every one of these stations at once.

        The iron recipe needs a table *and* a furnace, and walking to one and then
        the other lands you next to whichever you visited second. If the two were
        placed apart there may be no such cell at all — which is what `craft` falls
        back on building for.
        """
        grid = self.grid()
        goal = self.walkable(grid) | self.diggable(grid)
        for station in stations:
            want = self.kind(station)
            near = np.zeros_like(want)
            for dr, dc in DELTA.values():
                near |= np.roll(np.roll(want, -dr, axis=0), -dc, axis=1)
            goal &= near
        return goal

    def craft(self, token: str, stations: tuple[str, ...] = ("CRAFTING_TABLE",)) -> bool:
        """Stand where every station the recipe needs is in reach, then press it.

        The requirements are not restated here — the engine knows them, and a table
        of them written out would be a second copy to keep in step. If the press did
        not fire, the caller learns it from the achievement not appearing.
        """
        if not self.walk(self.beside_all(stations), adjacent=False):
            if not self.build(stations):
                return False
        return self.act(token)

    def build(self, stations: tuple[str, ...]) -> bool:
        """Put down whatever we are not already standing next to.

        Cheaper than walking back across a level, and the only way to be next to
        two stations that were placed apart. Costs two wood and one stone, which is
        why the route tops both up before the iron craft.
        """
        for station in stations:
            want = self.kind(station)
            if any(want[cell] for cell in self.neighbours()):
                continue
            if not self.place(self.STATIONS[station]):
                return False
        return True

    def place(self, token: str) -> bool:
        """Put a station down beside us, and stay where we are.

        There is no turn-in-place action: pressing a direction turns you only when
        the move is refused, and a station can only be placed on ground you *could*
        have walked onto. So facing the cell you want to build on walks you into it,
        and the naive version drifts a square every time it builds — which is how
        the table placed for an iron pickaxe ended up two squares from the furnace
        placed a moment later, with the recipe silently not firing.

        Step out and back instead: press away, press back, and you are where you
        started facing the cell you meant. Two actions, and both stations end up
        within reach of one square.
        """
        rows, cols = self.grid().shape
        free = self.walkable(self.grid())
        r, c = self.at()
        for token_to, (dr, dc) in DELTA.items():
            ahead, behind = (r + dr, c + dc), (r - dr, c - dc)
            if not all(0 <= v < n for v, n in
                       ((ahead[0], rows), (ahead[1], cols),
                        (behind[0], rows), (behind[1], cols))):
                continue
            if not (free[ahead] and free[behind]):
                continue
            back = next(k for k, d in DELTA.items() if d == (-dr, -dc))
            if not self.act(back) or self.at() != behind:
                return False
            if not self.act(token_to) or self.at() != (r, c):
                return False
            return self.act(token)
        return False


def run(game, budget: int) -> Router:
    """Walk the route as far as the budget allows.

    Ordered by dependency, and each step is allowed to fail: a level with no coal
    within reach is a fact about the seed, not a broken route, and the run carries
    on to whatever is still possible.
    """
    r = Router(game, budget)
    # Wood is spent by nearly everything — two for the table and one for each of
    # five recipes — so it is topped up before each craft rather than banked once.
    r.harvest("TREE", "wood", 4)
    r.place("place_table")
    r.harvest("TREE", "wood", 2)
    r.craft("make_wood_pickaxe")
    r.craft("make_wood_sword")
    r.upkeep()
    r.harvest("STONE", "stone", 4)
    r.harvest("TREE", "wood", 3)
    r.craft("make_stone_pickaxe")
    r.craft("make_stone_sword")
    r.upkeep()
    r.place("place_furnace")
    r.harvest("COAL", "coal", 1)
    r.upkeep()
    r.harvest("IRON", "iron", 1)
    # Enough to rebuild a table and a furnace beside the iron if the ones placed
    # earlier are half a level away, and still have the recipe's own wood and stone.
    r.harvest("TREE", "wood", 4)
    r.harvest("STONE", "stone", 3)
    r.upkeep()
    r.craft("make_iron_pickaxe", ("CRAFTING_TABLE", "FURNACE"))
    r.craft("make_iron_sword", ("CRAFTING_TABLE", "FURNACE"))
    return r
