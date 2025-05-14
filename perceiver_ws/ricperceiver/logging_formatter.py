import logging

# Definiamo i codici ANSI per i colori
RESET   = "\033[1;0m"
BLUE    = "\033[1;84m"
GREEN   = "\033[1;92m"
YELLOW  = "\033[1;93m"
RED     = "\033[1;91m"
WHITE   = "\033[1;97m"

# Formatter personalizzato per iniettare i colori basati sul livello
class ColoredFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: BLUE,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, RESET)
        name = f"{WHITE}[{record.name}]{RESET}"
        time = f"{WHITE}[{self.formatTime(record, '%H:%M:%S')}]{RESET}"
        level = f"{color}[{record.levelname}]{RESET}"
        message = f"{color}{record.getMessage()}{RESET}"

        return f"{name} {time} {level} {message}"
