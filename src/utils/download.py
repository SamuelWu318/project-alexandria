from utils.core import Path, ProjectPaths as proj
from utils.core import subprocess, time, zipfile
from utils import log

def run(ids: int | list[int], force: bool = False, rebuild: bool = True, attempts: int = 3):
    # ensure list type, ensure items in list
    if isinstance(ids, int): ids = [ids]
    if not ids: log.skip("no book ids were provided"); return

    # ensure zipfile directory exists
    proj.ZIPFILES_DIR.mkdir(parents=True, exist_ok=True)
    id_files = id_and_files(ids)
    subprocesses = {}
    rebuild_ids = []
    
    for id, file in id_files.items():
        # check if file exists, skip if not forced, otherwise delete
        if file.is_file() and not force: log.skip("book %d already exists", id); continue
        if file.is_file(): log.debug("book %d unlinked for replacement", id); file.unlink()

        log.step("Book ID: %d | BEGIN DOWNLOADING", id)
        cmd = [
            "wget", "-nd", "-q", "--no-check-certificate",
            f"https://aleph.gutenberg.org/cache/epub/{id}/pg{id}-h.zip"
        ]
        subprocesses[id] = subprocess.Popen(cmd, cwd=proj.ZIPFILES_DIR)

        log.debug("command %s ran", " ".join(cmd))
        time.sleep(2) # follow recommended time between downloads

    for id, subp in subprocesses.items():
        code = subp.wait()
        if code == 0: log.success("Book ID: %d | DOWNLOAD COMPLETE", id)
        else: log.error("Book ID: %d | DOWNLOAD ERROR (exit %d)", id, code)

    for id, file in id_files.items():
        ensured = ensure_zip(file, id, rebuild) # always log zip validation
        if rebuild and not ensured: rebuild_ids.append(id)
        
    if rebuild and rebuild_ids and attempts > 1:
        log.debug("rebuilding %d ids", len(rebuild_ids))
        run(rebuild_ids, True, rebuild, attempts=attempts-1)


def id_and_files(ids: list[int]) -> dict[int, Path]:
    ret = {}
    for id in ids:
        # skip ids if they are empty
        if not id: log.skip("a given book id was empty"); continue
        ret[id] = proj.ZIPFILES_DIR / f"pg{id}-h.zip"

    return ret


def ensure_zip(file: Path, id: int, rebuild: bool) -> bool:
    # check if missing entirely
    if not zipfile.is_zipfile(file):
        log.warning("Book ID: %d | MISSING ZIP, REBUILD? %s", id, rebuild)
        return False

    # check if zip corrupted
    with zipfile.ZipFile(file) as zfile:
        if zfile.testzip() is None: 
            log.success("Book ID: %d | DOWNLOAD ENSURED", id)
            return True
        else:
            log.warning("Book ID: %d | DOWNLOAD CORRUPTED, REBUILD? %s", id, rebuild) 
            return False