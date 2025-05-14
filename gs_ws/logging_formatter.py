import logging

# Definiamo i codici ANSI per i colori
class Colors:
    RESET   = "\033[1;0m"
    BLUE    = "\033[1;94m"
    GREEN   = "\033[1;92m"
    YELLOW  = "\033[1;93m"
    RED     = "\033[1;91m"
    PURPLE  = "\033[1;95m"
    WHITE   = "\033[1;97m"
    RED_BG  = "\033[0;101m"

# Formatter personalizzato per iniettare i colori basati sul livello
class ColoredFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: Colors.BLUE,
        logging.INFO:  Colors.WHITE,
        logging.WARNING:  Colors.YELLOW,
        logging.ERROR:  Colors.RED,
        logging.CRITICAL:  Colors.RED,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
        name = f"{color}[{record.name}]{Colors.RESET}"
        time = f"{color}[{self.formatTime(record, '%H:%M:%S')}]{Colors.RESET}"
        level = f"{color}[{record.levelname}]{Colors.RESET}"
        if record.levelname == "INFO":
            message = record.getMessage()
        else:
            message = f"{color}{record.getMessage()}{Colors.RESET}"
        return f"{name} {time} {level} {message}"
