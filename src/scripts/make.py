import os,sys

def build(num):
        os.system('make cleanreally')
        os.system('make cleanreally')
	make_command='make -j' + str(num)
	make=os.system(make_command)

	if make > 0:
               sys.exit(1) 
               
        return make

machine_name = os.uname()[1]

if machine_name == 'inferno':
	os.system('../src/scripts/pump_make.sh')
else:
	build(sys.argv[1])

os.system('make link_inputs')

exit
