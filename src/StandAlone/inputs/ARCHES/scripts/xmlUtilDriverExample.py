from xmlUtil import UPSxmlHelper as upshelp
from xmlUtil import UPSNodeObject as upsnode

my_node = upsnode("my_great_new_node", attributes=['label'], att_values=['somelabel'], children=['child1','emiss_coef', 'child3'], children_values=['some valeu','[0.15232498, 0.52384151, 0.09984559, -1.57580052, 1.30442371, -0.04325339, -0.06112965, 0.11475969, 0.15905172, -0.20896006, 0.05516192, 0.02045163, 0.02851480, 0.08187140, -0.13939356, 0.04270075, -0.02222587, 0.04199073, -0.07856715, 0.03076408, 0.02385809, -0.02312147, 0.01466296, 0.00639124, -0.00019175, 0.63713088, -0.00726427, -0.30457445]','I will be deleted'])
great_subnode = upsnode("jello_world", value='jello!')

my_ups = upshelp("../helium_1m.ups")

my_ups.add_node('./CFD/ARCHES', my_node)

my_ups.add_node('./CFD/ARCHES/my_great_new_node',great_subnode)

my_ups.change_node_value('./CFD/ARCHES/my_great_new_node/child1', 'oops..I mean value')

my_ups.rm_node('./CFD/ARCHES/my_great_new_node/child3')

my_ups.save_ups("mod_helium_1m.ups")
