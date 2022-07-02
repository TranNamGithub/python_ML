import msvcrt
import time

def readch():
    """ Get a single character on Windows.
 
    see https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/getch-getwch?view=vs-2019
    """
    ch = msvcrt.getch()
    if ch in b'\x00\xe0':  # Arrow or function key prefix?
        ch = msvcrt.getch()  # Second call returns the actual key code.
    return ch
while True:
    print(1)
    if msvcrt.kbhit():
        key = ord(readch())
        print(key)
        if key == 98:  # ord('a')
            print("la")
        elif key == 27:  # Escape key?
            break
print('Done')  