import numpy as np
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('file not ready to be read')
    else:
        array = np.load(sys.argv[1])
        np.savetxt(sys.argv[1][:-4]+".csv", array, delimiter=",")
        print(array)
