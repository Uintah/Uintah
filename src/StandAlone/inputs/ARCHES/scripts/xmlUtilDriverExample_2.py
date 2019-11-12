from xmlUtil import UPSxmlHelper as upshelp
from xmlUtil import UPSNodeObject as upsnode
import os

#In this example we want to find all files in the input file directory 
# that have nodes meeting a specific criteria: 
#
# <src type="do_radiation"...>
#    <DORadiationModel type="sweepSpatiallyParallel">
#

my_node = './CFD/ARCHES/TransportEqns/Sources'
my_atts = ['type']
my_att_values = ['do_radiation']

directory = '.'

print('These files match the criteria: ')

for subdir, dirs, files in os.walk(directory):
  for filename in files:

    filepath = subdir + os.sep + filename

    if filepath.endswith(".ups"):

      my_ups = upshelp(filepath)

      #get the src entry: 
      radiation_src = my_ups.get_node_subnodes(my_node, 'src', my_atts, my_att_values)
      
      #get the solver: 
      if radiation_src is not None: 

        model = (radiation_src.find('DORadiationModel').attrib)['type']

        if model == 'sweepSpatiallyParallel': 

          print(filepath)


