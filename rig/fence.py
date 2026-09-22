#!/usr/bin/env python3
"""Run a command with the filesystem it is allowed to see, and nothing else.

A session plays by discovering the rules, and everything it would otherwise have to
discover is on this disk: the package's own source carries the world generator, the
block table and all 67 achievements by name, and this harness carries a scripted
player that walks the early tech tree. Neither is a secret from the operating system.
Both are one `rg` away.

The Claude arm never reached for either across thirteen sessions and thirty thousand
actions, so the harness had no reason to stand anywhere but the audit. The Codex arm
reached in its first smoke run: eight actions played, then `tools/route.py`,
`tools/baselines.py` and the package's constants. The audit caught it — that is what
the audit is for — but catching it at the end of a fifteen-hour run yields a score
that measures reading rather than playing.

**Why Landlock and not the CLI's own sandbox.** Codex sandboxes through bubblewrap,
which needs an unprivileged user namespace, and this machine sets
`kernel.apparmor_restrict_unprivileged_userns=1`; without root that is the end of it
(see `Codex` in agents.py, where a session whose every command died still answered
the prompt from what it expected them to produce). Landlock is the one fence a
process may put around *itself*: no privilege, no namespace, no helper. It is applied
here and then `exec`ed through, and the kernel carries it into every child — the
shell a session opens, the `./act` it runs, the script it writes.

**It is an allowlist, and that shapes everything below.** Landlock has no deny rule.
Hiding one directory means naming every sibling that stays, which is why the caller
passes the harness's interpreter as its contents rather than as itself. A path that
is not named is not reachable, so the failure mode is a session that cannot work
rather than one that quietly still can: fenced too tight is loud, fenced too loose is
silent, and the whole point is not to be silently loose.

**What it does not do.** Not the network — Landlock ABI 4 can restrict TCP but the
session needs its own API and nothing here would be improved by a port list, and the
audit already counts web requests. Not `/proc`, which is granted whole: a session
that walks it can see this launcher's own command line, as it could before. This
fences reading the answer off the disk, which is the thing that was actually
happening.
"""

import ctypes
import os
import sys
from collections.abc import Iterable
from pathlib import Path

# x86_64. The three syscalls are the whole interface: ask the kernel what it
# supports, build a ruleset, then hand yourself to it.
CREATE_RULESET = 444
ADD_RULE = 445
RESTRICT_SELF = 446
VERSION = 1 << 0
PATH_BENEATH = 1
NO_NEW_PRIVS = 38  # prctl; restricting yourself is refused without it

# The access rights, by the ABI that introduced them. Everything through MAKE_SYM is
# ABI 1; REFER, TRUNCATE and IOCTL_DEV arrived later, and asking for a right the
# running kernel does not know is an error rather than a no-op — so `handled` below
# is built against what this kernel actually reports.
EXECUTE, WRITE_FILE, READ_FILE, READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
REMOVE_DIR, REMOVE_FILE = 1 << 4, 1 << 5
MAKE_CHAR, MAKE_DIR, MAKE_REG, MAKE_SOCK = 1 << 6, 1 << 7, 1 << 8, 1 << 9
MAKE_FIFO, MAKE_BLOCK, MAKE_SYM = 1 << 10, 1 << 11, 1 << 12
REFER, TRUNCATE, IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15

ABI_BITS = {1: (1 << 13) - 1, 2: REFER, 3: TRUNCATE, 5: IOCTL_DEV}
# Rights that only mean something about a directory. Adding one to a rule whose path
# is a regular file is EINVAL, and the credential and the two scripts below are
# files, so a rule's access is masked by what its path can hold.
FILE_ONLY = EXECUTE | WRITE_FILE | READ_FILE | TRUNCATE | IOCTL_DEV

READ = EXECUTE | READ_FILE | READ_DIR
WRITE = (
    WRITE_FILE | REMOVE_DIR | REMOVE_FILE | MAKE_CHAR | MAKE_DIR | MAKE_REG
    | MAKE_SOCK | MAKE_FIFO | MAKE_BLOCK | MAKE_SYM | REFER | TRUNCATE | IOCTL_DEV
)


class RulesetAttr(ctypes.Structure):
    _fields_ = (("handled_access_fs", ctypes.c_uint64),
                ("handled_access_net", ctypes.c_uint64))


class PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = (("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32))


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def _fail(what: str) -> None:
    err = ctypes.get_errno()
    raise SystemExit(f"fence: {what} failed: {os.strerror(err)} ({err})")


def supported() -> int:
    """This kernel's Landlock ABI, or 0 if it has none.

    Asked by passing no ruleset at all, which is the documented way to version-check
    and the only call here that is allowed to come back empty-handed.
    """
    libc = _libc()
    libc.syscall.restype = ctypes.c_long
    libc.syscall.argtypes = (ctypes.c_long, ctypes.c_void_p, ctypes.c_size_t,
                             ctypes.c_uint32)
    abi = libc.syscall(CREATE_RULESET, None, 0, VERSION)
    return max(abi, 0)


def apply(readable: Iterable[Path], writable: Iterable[Path],
          listed: Iterable[Path] = ()) -> int:
    """Restrict this process to these paths, and return the ABI it was done under.

    Every path is a hierarchy: granting a directory grants everything beneath it,
    which is why the caller hands over the contents of a directory it wants to open
    only partly. Nothing outside is readable afterwards, including through a symlink
    that points out — the kernel resolves the target and checks that, so the
    credential this harness symlinks into each session's config directory has to be
    granted by name.

    **Rules add, they never subtract.** A rule on a directory inside a granted one
    widens it; there is no way to narrow it. So a tree that must be open except for
    one thing is passed as its contents minus that thing, and the thing is simply
    never named. The kernel rejects a rule granting nothing (ENOMSG), which is the
    same statement from the other side: you cannot write a hole.

    `listed` is the exception that makes the rest work. A directory granted this way
    can be read as a list of names and not as files, which is exactly what Python's
    import machinery needs of the directory a script lives in — it lists it to find
    the module beside it. Without it a fenced `./act` could not import what it sits
    next to; with it, the harness's directory gives up its filenames and none of its
    contents.
    """
    abi = supported()
    if not abi:
        raise SystemExit(
            "fence: this kernel has no Landlock — a session would run unfenced and "
            "nothing would say so. Refusing to start."
        )
    handled = 0
    for level, bits in ABI_BITS.items():
        if abi >= level:
            handled |= bits

    libc = _libc()
    libc.syscall.restype = ctypes.c_long
    libc.syscall.argtypes = (ctypes.c_long, ctypes.c_void_p, ctypes.c_size_t,
                             ctypes.c_uint32)
    attr = RulesetAttr(handled_access_fs=handled, handled_access_net=0)
    # ABI 4 added the network field. An older kernel is handed the older struct,
    # because a size it does not know is an error rather than a longer struct.
    size = ctypes.sizeof(attr) if abi >= 4 else ctypes.sizeof(ctypes.c_uint64)
    ruleset = libc.syscall(CREATE_RULESET, ctypes.byref(attr), size, 0)
    if ruleset < 0:
        _fail("creating the ruleset")

    libc.syscall.argtypes = (ctypes.c_long, ctypes.c_int, ctypes.c_uint32,
                             ctypes.c_void_p, ctypes.c_uint32)
    for paths, access in ((readable, READ), (writable, READ | WRITE),
                          (listed, READ_DIR)):
        for path in paths:
            if not path.exists():
                raise SystemExit(
                    f"fence: {path} does not exist. Every path is named on purpose, "
                    f"so a missing one is a mistake in the allowlist rather than "
                    f"something to skip."
                )
            # O_PATH: the fence describes a place, and opening it for real would
            # need the very access being granted.
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                allowed = access & handled
                if not path.is_dir():
                    allowed &= FILE_ONLY
                rule = PathBeneathAttr(allowed_access=allowed, parent_fd=fd)
                if libc.syscall(ADD_RULE, ruleset, PATH_BENEATH,
                                ctypes.byref(rule), 0) < 0:
                    _fail(f"granting {path}")
            finally:
                os.close(fd)

    libc.prctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                           ctypes.c_ulong, ctypes.c_ulong)
    if libc.prctl(NO_NEW_PRIVS, 1, 0, 0, 0) < 0:
        _fail("setting no_new_privs")
    libc.syscall.argtypes = (ctypes.c_long, ctypes.c_int, ctypes.c_uint32)
    if libc.syscall(RESTRICT_SELF, ruleset, 0) < 0:
        _fail("restricting this process")
    os.close(ruleset)
    return abi


def main(argv: list[str]) -> int:
    readable: list[Path] = []
    writable: list[Path] = []
    listed: list[Path] = []
    rest: list[str] = []
    tokens = iter(argv)
    for token in tokens:
        if token == "--":
            rest = list(tokens)
            break
        target = {"--ro": readable, "--rw": writable, "--ls": listed}.get(token)
        if target is None:
            raise SystemExit(f"fence: {token} is not --ro, --rw, --ls or --")
        try:
            target.append(Path(next(tokens)).resolve())
        except StopIteration:
            raise SystemExit(f"fence: {token} needs a path") from None
    if not rest:
        raise SystemExit(
            "fence: usage: fence.py [--ro DIR] [--rw DIR] [--ls DIR] -- COMMAND [ARG ...]"
        )
    apply(readable, writable, listed)
    # Not a subprocess: the point is that what runs next *is* this process, so there
    # is no unfenced parent left holding a descriptor to somewhere it should not.
    os.execvp(rest[0], rest)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
