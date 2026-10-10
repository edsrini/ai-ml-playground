import logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./results.log", mode="w"),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

def days_until_launch(current_day, launch_day):
    """"Returns the days left before launch.

    current_day (int) - current day in integer
    launch_day (int) - launch day in integer
    """
    logging.info("current_day: %s", current_day)
    return launch_day - current_day