import os

def last_input_path(target):

    last_path = os.path.basename(os.path.normpath(target))
    return last_path

target = 'carlo/beretta/N/P/Casa'
last_path = last_input_path(target)
print(last_path)