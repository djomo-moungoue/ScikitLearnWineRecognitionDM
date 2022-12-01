from pathlib import Path
from datetime import datetime


class LoggerZa:
    CURRENT_FILE = Path(__file__).resolve()
    BASE_DIR = CURRENT_FILE.parent
    name = "LoggerZa"
    counter = 0

    def __init__(self):
        pass

    @classmethod
    def log(cls, message: str, to_console=True, to_file=True, txt_format=True) -> str:
        """
        message: str\n
        to_console=True | False. Is default True\n
        to_file=True | True. Is default True\n
        txt_format=True | True. Is default True. csv format when it is false\n
        """
        cls.counter += 1
        if to_console:
            if txt_format:
                print(f"{message}")
            else:
                print(f"{cls.counter}, {datetime.now()}, {message}")
        if to_file:
            log_dir = cls.BASE_DIR / "logs"
            if not Path(log_dir).exists():
                print(f"log_dir : {log_dir}")
                Path(log_dir).mkdir()
            log_file = cls.BASE_DIR / "logs/log_za.csv"
            if txt_format:
                if not Path(log_file).exists():
                    print(f"log_file : {log_file}")
                    with open(log_file, "a") as log_append:
                        log_append.write(f"Logging Message\n")
                        if cls.counter == 1:
                            log_append.write(f"\n-----------------------{datetime.now()}------------------------------------\n")
                else:
                    with open(log_file, "a") as log_append:
                        if cls.counter == 1:
                            log_append.write(f"\n-----------------------{datetime.now()}------------------------------------\n")
                        log_append.write(f"{message}\n")
            else:
                if not Path(log_file).exists():
                    print(f"{cls.counter}, {datetime.now()}, log_file : {log_file}")
                    with open(log_file, "a") as log_append:
                        log_append.write(f"ID, Object Name, Logging Message\n")
                else:
                    with open(log_file, "a") as log_append:
                        log_append.write(f"{cls.counter}, {datetime.now()}, {message}")

    @classmethod
    def validate_path(cls, file_path) -> bool:
        if not Path(file_path).exists():
            LoggerZa.log(f"File : {file_path} - NOT found")
            return False
        LoggerZa.log(f"File : {file_path} - found")
        return True


if __name__ == "__name__":
    LoggerZa.log(f"Abs Path to CURRENT_FILE : {LoggerZa.CURRENT_FILE}")
    LoggerZa.log(f"Abs Path to BASE_DIR : {LoggerZa.BASE_DIR}")
