from utils.core import ProjectPaths as proj
from utils.core import csv, re, defaultdict
from utils.core import log_return
from utils import log


CATALOG = {} # {id: {...}}
with proj.CATALOG.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f): CATALOG[row["Text#"]] = row

# --- get metadata pieces --- #

def contributors(id: int) -> dict[str, list[str]]:
    log.step("Book ID: %d | authors requested", id)
    ret = defaultdict(list)

    contributor_string = CATALOG[str(id)]["Authors"]
    if not contributor_string: return {"Authors": ["unknown"]}

    contributors = [c.strip() for c in contributor_string.split(";")]
    for contributor in contributors:
        parts, num = re.subn(r"\[.*?\]", "", contributor)

        if num: # get role and make it plural
            role = re.findall(r"\[.*?\]", contributor)[0].strip('[').strip(']')
            role = role.replace(" ", "s ", 1) if " " in role else role + "s"
        else: role = "Authors"

        parts = [part.strip() for part in parts.split(",")]
        name = []

        for part in parts: # reorder all parts that contain no numbers
            if re.search(r"\d", part): continue
            name.insert(0, part)

        ret[role].append(" ".join(name).strip())

    return log_return(id, "contributors", dict(ret))


def title(id: int) -> str:
    log.step("Book ID: %d | title requested", id)
    return log_return(id, "title", CATALOG[str(id)]["Title"])


def subjects(id: int):
    log.step("Book ID: %d | subjects requested", id)
    subjects = CATALOG[str(id)]["Subjects"].split(";")
    return log_return(id, "subjects", [subject.strip() for subject in subjects])


def language(id: int):
    log.step("Book ID: %d | language requested", id)
    return log_return(id, "language", CATALOG[str(id)]["Language"])


def raw_html(id: int):
    log.step("Book ID: %d | raw html requested", id)
    