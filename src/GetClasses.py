#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************************
* GetClasses
* version 2025.03.07.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************************
'''
print(__doc__)

import sys, os.path, pickle
from libDataML import Normalizer

#***************************************************
''' This is needed for installation through pip '''
#***************************************************
def GetClasses():
    main()

#************************************
# Main
#************************************
def main():
    if len(sys.argv)<3:
        usage()
        sys.exit(2)

    print("\n Opening files:",sys.argv[1],"and",sys.argv[2])
    try:
        #norm = pickle.loads(open(sys.argv[1], "rb").read())
        #le = pickle.loads(open(sys.argv[2], "rb").read())
        norm_file = open(open(sys.argv[1], "rb"))
        norm = pickle.loads(norm_file.read())
        norm_file.close()
        le_file = open(sys.argv[2], "rb")
        le = pickle.loads(le_file.read())
        le_file.close()
        
        print("\n Classes: ",le.classes_())

        normText = []
        for i in le.classes_():
            #normText += "{0:.2f}  {1:.2f}\n".format(i,norm.transform_inverse_single(i))
            normText.append(norm.transform_inverse_single(i))
        print("\n Normalized classes: \n",normText)

        with open(os.path.splitext(sys.argv[1])[0]+".txt", "w+") as text_file:
            text_file.write(str(normText))
        print("\n Classes saved in:",os.path.splitext(sys.argv[1])[0]+".txt\n")
    except:
        print("\n Invalid/missing File \n")

#************************************
# Lists the program usage
#************************************
def usage():
    print('\n Usage:\n')
    print('  python3 GetClasses.py -t <pkl normalization file> <pkl MultiClass file>\n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
