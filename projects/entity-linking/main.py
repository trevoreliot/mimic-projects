import logging
from src.utils import log_hardware_status

# Configure logging to output INFO statements
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def main():
    print("Starting Clinical Entity Linking Pipeline POC...")
    log_hardware_status()


if __name__ == "__main__":
    main()

