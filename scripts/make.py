import os,sys

def build(num):
        os.system('make cleanreally')
	make_command='make -j' + str(num)
	make=os.system(make_command)

	if make > 0:
               sys.exit(1) 
               
        return make

build(sys.argv[1])

os.system('make link_inputs')

exit
