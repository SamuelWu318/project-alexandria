from utils import log, download
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
    print(data.get.authors(i))
    print(data.get.title(i))
    print(data.metadata.build(i))
    
