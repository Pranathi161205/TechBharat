import os
import time
import logging
import glob
import job_worker  # reuse your existing job_worker.py functions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [daemon] %(levelname)s %(message)s"
)

JOBS_DIR = os.path.join(os.path.dirname(__file__), "jobs")

def main():
    logging.info("Starting job worker daemon (watching %s)", JOBS_DIR)
    os.makedirs(JOBS_DIR, exist_ok=True)

    while True:
        # find all jobs waiting to be processed
        for job_file in glob.glob(os.path.join(JOBS_DIR, "*.json")):
            try:
                logging.info("Found job %s", job_file)
                # run the same processor you already have in job_worker.py
                job_worker.process_job(job_file)
            except Exception as e:
                logging.exception("Error processing %s: %s", job_file, e)
        time.sleep(5)  # check every 5 seconds

if __name__ == "__main__":
    main()
