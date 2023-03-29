#!/usr/bin/env python3

def writeDividerLine( filename ):
    file = open( filename, 'a')
    file.write("______________________________________________________________________\n")
    file.close  


def appendFile( filename, msg ):
    file = open(filename, 'a')
    file.write( msg )
    file.close()
