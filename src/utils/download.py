import subprocess, time
import zipfile

from pathlib import Path
from utils.core import ProjectPaths as proj
from utils import log

def _ensure_zip(file: Path, id: int, rebuild: bool) -> bool:
    # check if missing entirely
    if not zipfile.is_zipfile(file):
        log.warning("Book ID: %d | missing zip, rebuild? %s", id, rebuild)
        return False

    # check if zip corrupted
    with zipfile.ZipFile(file) as zfile:
        if zfile.testzip() is None: 
            log.success("Book ID: %d | download ensured", id)
            return True
        else:
            log.warning("Book ID: %d | download corrupted, rebuild? %s", id, rebuild) 
            return False


def zip(ids: int | list[int], force: bool = False, rebuild: bool = True, attempts: int = 3):
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

        log.step("Book ID: %d | begin downloading", id)
        cmd = [
            "wget", "-nd", "-q", "--no-check-certificate",
            f"https://aleph.gutenberg.org/cache/epub/{id}/pg{id}-h.zip"
        ]
        subprocesses[id] = subprocess.Popen(cmd, cwd=proj.ZIPFILES_DIR)

        log.debug("command %s ran", " ".join(cmd))
        time.sleep(2) # follow recommended time between downloads

    for id, subp in subprocesses.items():
        code = subp.wait()
        if code == 0: log.success("Book ID: %d | download complete", id)
        else: log.error("Book ID: %d | download error (exit %d)", id, code)

    for id, file in id_files.items():
        ensured = _ensure_zip(file, id, rebuild) # always log zip validation
        if rebuild and not ensured: rebuild_ids.append(id)
        
    if rebuild and rebuild_ids and attempts > 1:
        log.debug("rebuilding %d ids", len(rebuild_ids))
        zip(rebuild_ids, True, rebuild, attempts=attempts-1)


def id_and_files(ids: list[int]) -> dict[int, Path]:
    ret = {}
    for id in ids:
        # skip ids if they are empty
        if not id: log.skip("a given book id was empty"); continue
        ret[id] = proj.ZIPFILES_DIR / f"pg{id}-h.zip"

    return ret

def _html(zip_files: list[Path], force: bool = False) -> list[Path]:
    if not zip_files: log.skip("no zipfile paths provided"); return
    bad_zip = []
    ret_paths = []
    for zip_file in zip_files:
        if not zip_file.exists(): 
            log.skip("no path %s exists", zip_file);
            bad_zip.append(zip_file); continue
        if not zip_file.is_file(): 
            log.skip("%s is a directory, not a file", zip_file);
            bad_zip.append(zip_file); continue

        dest_name = f"{zip_file.name[:-6]}.html"
        dest_path = proj.HTML_DIR / dest_name
        if dest_path.exists() and not force: continue

        zf = zipfile.ZipFile(zip_file, "r")
        for file in zf.filelist:
            if file.filename.endswith("/"): continue
            if file.filename.endswith(".html"):
                file.filename = dest_name
                zf.extract(file, proj.HTML_DIR)
                ret_paths = dest_path; break
        zf.close()

    if bad_zip: log.warning("%s: these zips caused issues", bad_zip)
    return ret_paths
    
def html(ids: int | list[int], force: bool = False) -> list[Path]:
    if isinstance(ids, int): ids = [ids]
    if not ids: log.skip("no book ids were provided"); return

    clean_ids = []
    for id in ids:
        if not id: log.skip("a given book id was empty"); continue
        if not isinstance(id, int): log.skip("a given book id was not an int"); continue
        clean_ids.append(id)

    zip_files_list = [proj.ZIPFILES_DIR / f"pg{id}-h.zip" for id in clean_ids]
    return _html(zip_files_list, force)
