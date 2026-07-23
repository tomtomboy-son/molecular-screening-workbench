import logging
import logging.config
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

logger = logging.getLogger(__name__)
# logger.debug("So googbye days.")
# logger.error("Wow, that's a big deal!")

print(logger)