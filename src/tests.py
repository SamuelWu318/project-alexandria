from utils import log, download
from utils.core import ProjectPaths as proj
import data

log.debug("test")
log.skip("test")
log.step("test")
log.info("test")
log.warning("test")
log.success("test")
log.error("test")
log.critical("test")

for i in range(1, 5):
    print(data.get.contributors(i))
    print(data.get.title(i))
    print(data.metadata.build(i))

download.zip([84, 4300, 103])
download.html([84, 4300, 103])