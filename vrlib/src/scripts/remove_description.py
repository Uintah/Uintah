# Python script for removing the first description tag from xml files.
#
# usage 'python remove_description.py foo.xml > new.xml

import sys, regex, string, regsub

rex_start = regex.compile('.*<description>.*');
rex_end = regex.compile('.*</description>.*');

def cont(file, line):
    while 1:
        n = rex_end.match(line);
        if n >= 0:
            break
        line = file.readline();
        if not line: break

def gen_newfile(filename):
    file = open(filename, 'r')
    first = 1
    while 1:
        line = file.readline()
        if not line: break

        n = rex_start.match(line)
        if n >= 0 and first:
            cont(file, line);
            first = 0
            continue

        print line,

if __name__ == '__main__':
    import sys
    gen_newfile(sys.argv[1])
        
