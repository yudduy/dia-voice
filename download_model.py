# download_model.py
# Utility script to download the Dia model and dependencies without starting the server.

import logging
import os
import engine  # Import the engine module to trigger its loading logic

# Configure basic logging for the script
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ModelDownloader")

if __name__ == "__main__":
    logger.info("--- Starting Dia Model Download ---")

    # Ensure cache directory exists (redundant if engine.load_model does it, but safe)
    try:
        from config import get_model_cache_path

        cache_path = get_model_cache_path()
        os.makedirs(cache_path, exist_ok=True)
        logger.info(
            f"Ensured model cache directory exists: {os.path.abspath(cache_path)}"
        )
    except Exception as e:
        logger.warning(f"Could not ensure cache directory exists: {e}")

    # Trigger the model loading function from the engine
    logger.info("Calling engine.load_model() to initiate download if necessary...")
    success = engine.load_model()

    if success:
        logger.info("--- Model download/load process completed successfully ---")
    else:
        logger.error(
            "--- Model download/load process failed. Check logs for details. ---"
        )
        exit(1)  # Exit with error code

    logger.info("You can now start the server using 'python server.py'")
