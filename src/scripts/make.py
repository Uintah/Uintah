import os,sys

def build(num):

	make_command='make -j' + str(num)
	make=os.system(make_command)

	if make > 0:
		os.system('make cleanreally')
		make=os.system(make_command)


	return make

build(sys.argv[1])

os.system('make link_inputs')

exit
