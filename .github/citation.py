"""
Write CITATION.cff, the file behind GitHub's "Cite this repository" button,
from the people in the git history.

The authors are the MAX_AUTHORS people with the most commits. A person
counts once for each commit that they authored or appear in as a
Co-authored-by trailer. .mailmap at the repo root merges the names and
emails of one person. AI tools (the rules of ai_credit.py) and bots are
not authors.

Usage, from the repo root, when the top contributors change:
    python3 .github/citation.py > CITATION.cff
"""

import json
import re
import subprocess
import sys
from collections import Counter

import ai_credit

TITLE = "NeoRacer MuJoCo"
ABSTRACT = (
    "A MuJoCo simulation of the NeoRacer, a 1/12-scale autonomous racing car, with its sensors, "
    "tracks, a procedural track generator and batched stepping on CPU or GPU."
)
REPOSITORY = "https://github.com/Neobotics-Foundation-Inc/neoracer-mujoco"
LICENSE = "GPL-3.0-only"

COAUTHOR = "Co-authored-by: "
MAX_AUTHORS = 5
HAS_EMAIL = re.compile(r" <[^<>]*>$")


def is_person(identity: str) -> bool:
    """True if identity ("Name <email>") is a person, not an AI tool or a bot."""
    name, _, rest = identity.partition(" <")
    email = rest.removesuffix(">")
    bot = "[bot]" in identity or email in ("noreply@github.com", "")
    return not bot and not ai_credit.credits_ai(COAUTHOR + identity) and bool(name)


def ranked_people(commits: list[list[str]]) -> list[str]:
    """The identities ("Name <email>") of the people credited in commits, the
    most credited commits first. Each commit is the list of identities that
    it credits. A person counts once per commit, and two identities with one
    name are two people. A tie keeps the order of the first credit."""
    people = [list(dict.fromkeys(i for i in c if is_person(i))) for c in commits]
    counts = Counter(i for c in people for i in c)
    return sorted(dict.fromkeys(i for c in people for i in c), key=lambda i: -counts[i])


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


def citation(people: list[str]) -> str:
    """The text of CITATION.cff, with the first MAX_AUTHORS of people
    ("Name <email>", most commits first) as the authors. Two entries of one
    list must differ, so an author whose entry would equal another one's
    also gets the email."""
    top = people[:MAX_AUTHORS]
    entries = [cff_person(p.partition(" <")[0]) for p in top]
    shared = Counter(entries)
    authors = ""
    for entry, person in zip(entries, top, strict=True):
        authors += entry
        if shared[entry] > 1:
            email = person.partition(" <")[2].removesuffix(">")
            authors += f"    email: {json.dumps(email)}\n"
    return (
        "# Written by .github/citation.py from the git history. Do not edit.\n"
        f"# The authors are the {MAX_AUTHORS} people with the most commits, counting\n"
        "# the commits that a person authored or co-authored.\n"
        "cff-version: 1.2.0\n"
        'message: "If you use this software, please cite it as below."\n'
        f"title: {json.dumps(TITLE)}\n"
        f"abstract: {json.dumps(ABSTRACT)}\n"
        "type: software\n"
        f"repository-code: {json.dumps(REPOSITORY)}\n"
        f"license: {LICENSE}\n"
        "authors:\n" + authors
    )


def git(*args: str, stdin: str = "") -> str:
    return subprocess.run(
        ["git", *args], input=stdin, capture_output=True, text=True, check=True
    ).stdout


def identities() -> list[list[str]]:
    """The identities that each commit on the current branch credits, its
    author and co-authors, oldest commit first, after .mailmap."""
    log = git(
        "log",
        "--reverse",
        "--format=%x1e%aN <%aE>%n%(trailers:key=Co-authored-by,valueonly)",
    )
    commits = [
        [line.strip() for line in c.splitlines() if line.strip()]
        for c in log.split("\x1e")
        if c.strip()
    ]
    # Trailers are not mapped by --use-mailmap, so map every identity.
    # check-mailmap takes only "Name <email>". A credit without an email, such
    # as "Co-authored-by: Claude", stays as it is, and is_person leaves it out.
    mappable = list(dict.fromkeys(i for c in commits for i in c if HAS_EMAIL.search(i)))
    stdin = "".join(f"{i}\n" for i in mappable)
    mapped = git("check-mailmap", "--stdin", stdin=stdin).splitlines()
    names = dict(zip(mappable, mapped, strict=True))
    return [[names.get(i, i) for i in c] for c in commits]


def main() -> int:
    sys.stdout.write(citation(ranked_people(identities())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
