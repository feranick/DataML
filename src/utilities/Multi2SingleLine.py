#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***************************************************
* Convert from multiline string into single line
* By: Nicola Ferralis <feranick@hotmail.com>
* version v2025.04.7.1
***************************************************
'''
#print(__doc__)

import sys, os.path, getopt

def main():

    saveInText = True
    
    if(len(sys.argv)<3):
        usage()
        return
    
    opts, args = getopt.getopt(sys.argv[1:],
        "tf:", ["text", "file"])
    
    for o, a in opts:
        if o in ("-t" , "--text"):
            string = sys.argv[2]
            print(string.replace('\n', ''))
    
        if o in ("-f" , "--file"):
            file = sys.argv[2]
            fileRoot = os.path.splitext(file)[0]
            outfile = fileRoot+"_single.txt"
    
            with open(file, 'r') as f:
                print(" Opening text file with multiple lines:",file,"\n")
                data = f.read().replace('\n', '')
                print(" Single line string: \n")
                print(data,"\n")
           
            if saveInText:
                with open(outfile, "w") as of:
                    of.write(data)
                    print(" Single line text file saved in:",outfile,"\n")
            
def usage():
    print(__doc__)
    print(' Usage:\n  python3 Multi2SingleLine.py -t \"<multi-line text>\"')
    print('  python3 Multi2SingleLine.py -f <file>')
    print('  python3 Multi2SingleLine.py -h\n')
            
#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
