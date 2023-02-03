import colorama

def init():
    colorama.init()

def section_print(message):
    print(colorama.Fore.GREEN + '==> ' + message + colorama.Style.RESET_ALL)

def subsection_print(message):
    print(colorama.Fore.BLUE + '====> ' + message + colorama.Style.RESET_ALL)
