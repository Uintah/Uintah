import os,sys

#______________________________________________________________________
#  This script is called by the buildbot machinery and compiles uintah
#______________________________________________________________________

def compileUintah(num):
    
    print "Executing: make clean twice"
    os.system('make clean')
    os.system('make clean')

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
