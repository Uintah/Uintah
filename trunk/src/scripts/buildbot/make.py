import os,sys

#______________________________________________________________________
#  This script is called by the buildbot machinery and compiles uintah
#______________________________________________________________________

def compileUintah(num):
    
    print "Executing: make cleanreally twice"
    os.system('make cleanreally')
    os.system('make cleanreally')

    print "Executing: make -j %s" % str(num)
    make_command='make -j' + str(num)
    make=os.system(make_command)

    if make > 0:
      sys.exit(1) 
      
    return make

#__________________________________
compileUintah(sys.argv[1])
os.system('make link_inputs')

exit
