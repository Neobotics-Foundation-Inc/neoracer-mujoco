"""
Write CITATION.cff, the file behind GitHub's "Cite this repository" button,
from the people in the git history.

The authors are every person who authored a commit or appears in a
Co-authored-by trailer, the most commits first. .mailmap at the repo root
merges the names and emails of one person. AI tools (the rules of
ai_credit.py) and bots are not authors.

Usage, from the repo root, after a new person's first commit:
    python3 .github/citation.py > CITATION.cff
"""

import json
import subprocess
import sys
from collections import Counter

import ai_credit

TITLE = "NeoRacer MuJoCo"
ABSTRACT = (
    "A MuJoCo simulation of the NeoRacer, a 1/12-scale RC car, with its sensors, "
    "tracks, a procedural track generator and batched stepping on CPU or GPU."
)
REPOSITORY = "https://github.com/Neobotics-Foundation-Inc/neoracer-mujoco"
LICENSE = "GPL-3.0-only"

COAUTHOR = "Co-authored-by: "


def is_person(identity: str) -> bool:
    """True if identity ("Name <email>") is a person, not an AI tool or a bot."""
    name, _, rest = identity.partition(" <")
    email = rest.removesuffix(">")
    bot = "[bot]" in identity or email in ("noreply@github.com", "")
    return not bot and not ai_credit.credits_ai(COAUTHOR + identity) and bool(name)


def ranked_names(identities: list[str]) -> list[str]:
    """The names of the people among identities, the most credits first. Two
    people with the same count keep the order of their first credit."""
    names = [i.partition(" <")[0] for i in identities if is_person(i)]
    counts = Counter(names)
    return sorted(dict.fromkeys(names), key=lambda n: -counts[n])


# Words that start a family name, as in Abdallah El Hamawi or Guido van Rossum.
PARTICLES = frozenset(
    {
        "al",
        "bin",
        "da",
        "de",
        "del",
        "der",
        "di",
        "du",
        "el",
        "ibn",
        "la",
        "le",
        "van",
        "von",
    }
)


def cff_person(name: str) -> str:
    """One entry of the authors list. A name of one word, such as a GitHub
    handle, is an alias. Otherwise the family name is the last word, with the
    particles before it."""
    words = name.split()
    if len(words) == 1:
        return f"  - alias: {json.dumps(name)}\n"
    start = len(words) - 1
    while start > 1 and words[start - 1].lower() in PARTICLES:
        start -= 1
    return (
        f"  - given-names: {json.dumps(' '.join(words[:start]))}\n"
        f"    family-names: {json.dumps(' '.join(words[start:]))}\n"
    )


def citation(names: list[str]) -> str:
    """The text of CITATION.cff for the authors names."""
    return (
        "# Written by .github/citation.py from the git history. Do not edit.\n"
        "cff-version: 1.2.0\n"
        'message: "If you use this software, please cite it as below."\n'
        f"title: {json.dumps(TITLE)}\n"
        f"abstract: {json.dumps(ABSTRACT)}\n"
        "type: software\n"
        f"repository-code: {json.dumps(REPOSITORY)}\n"
        f"license: {LICENSE}\n"
        "authors:\n" + "".join(cff_person(n) for n in names)
    )


def git(*args: str, stdin: str = "") -> str:
    return subprocess.run(
        ["git", *args], input=stdin, capture_output=True, text=True, check=True
    ).stdout


def identities() -> list[str]:
    """Every commit author and co-author on the current branch, oldest first,
    after .mailmap."""
    log = git(
        "log",
        "--reverse",
        "--format=%aN <%aE>%n%(trailers:key=Co-authored-by,valueonly)",
    )
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    return git("check-mailmap", "--stdin", stdin="\n".join(lines) + "\n").splitlines()


def main() -> int:
    sys.stdout.write(citation(ranked_names(identities())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
