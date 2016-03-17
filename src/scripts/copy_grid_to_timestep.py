import os
import time 
import xml.etree.ElementTree as ET
import xml.dom.minidom as MD
import xml.dom.pulldom as pulldom
import sys
from lxml import etree


## Use like this:
# python copy_grid_to_timestep.py arg1 arg2
# arg1 = uda_base_name (i.e. data.uda)
# arg2 = (optional with a default of '.' ) path to the directory containing the uda's (i.e. python /path/to/uda)
# python copy_grid_to_timestep.py data.uda /path/to/uda

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

# retrieve the arguments to the script
uda_base = sys.argv[1]
if len(sys.argv) == 2:
    directory = '.' 
if len(sys.argv) == 3:
    directory = sys.argv[2]
# first for loop over all files in the directory.
print "Searching for uda's with the base name: ",directory,"/",uda_base
str="%s." % (uda_base)
for uda in os.listdir(directory):
    if str in uda:
        print "Found valid .uda directory: ",uda
        str2="%s/%s" % (directory,uda)
        for tstep in os.listdir(str2):
            if "t" in tstep and hasNumbers(tstep):
                print "timestep: ",tstep
                grid_xml_str="%s/%s/%s/grid.xml" % (directory,uda,tstep)
                tstep_xml_str="%s/%s/%s/timestep.xml" % (directory,uda,tstep)
                orig_tstep_xml_str="%s/%s/%s/timestep.xml.orig" % (directory,uda,tstep)
                if os.path.exists(grid_xml_str) and os.path.exists(tstep_xml_str):
                    copy_str="cp %s %s" %(tstep_xml_str,orig_tstep_xml_str)
                    #TODO if file doesn't exist.. we need to move on to next loop.
                    parser = etree.XMLParser(remove_blank_text=True) 
                    newxml = etree.parse(tstep_xml_str, parser)
                    gridxml = etree.parse(grid_xml_str, parser)
                    root_newxml = newxml.getroot()
                    root_gridxml = gridxml.getroot()
                    write_grid_flag = True 
                    write_data_flag = True
                    for child in root_newxml:
                        if "Grid" in child.tag:
                            print "timestep.xml already contains Grid. Skipping."
                            write_grid_flag = False 
                        if "Data" in child.tag:
                            print "timestep.xml already contains Data. Skipping."
                            write_data_flag = False 
                    if write_grid_flag:
                        grid = root_gridxml.find('Grid')
                        root_newxml.append(grid)
                    if write_data_flag:
                        data = root_gridxml.find('Data')
                        root_newxml.append(data)
                    if write_grid_flag or write_data_flag:
                        print "writing timestep.xml."
                        # copy timestep.xml to timestep.xml.orig (if we are going to create a new timestep.xml)
                        os.system(copy_str)
                        newxml.write(tstep_xml_str, pretty_print=True)
                    else:
                        print "timestep.xml already contains both Grid and Data. Skipping."
