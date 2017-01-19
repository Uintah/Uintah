import sys
import os
import xml.etree.ElementTree as ET

def fix_ups(filename):

  tree = ET.parse(filename)
  root = tree.getroot()

  CFD = root.find('CFD')
  Arches = CFD.find('ARCHES')
  Properties = Arches.find('Properties')

  for n in Properties:

    Properties.remove(n)
    type=''
    if ( n.tag == 'ConstantProps'):
      type='constant'
    elif ( n.tag == "ClassicTable"):
      type='classic'
    else:
      type='coldflow'

    newTab = ET.Element('table')
    newTab.attrib['label'] = 'a_user_generated_label'
    newTab.attrib['type'] = type
    newTab.insert(0,n)
    Properties.insert(0,newTab)

  os.system('cp '+filename+' '+filename+'.orig_ups')

  tree.write(filename)

#------------------------------------------------------------

def usage():

  print 'Description: '
  print '  This script modifies all tables in the current directory from which you run the script '
  print '  with a user specified file extension. It updates the <Properties> node in the ups file '
  print '  to be compatible with the latest code by (if it currently is not) by adding a <table> '
  print '  node with a generic label (which you can change). '
  print '  Note that your original ups file is stored with a *.orig_ups extension.'
  print ''
  print 'Usage: '
  print '  python update_table_entry.py file_extension'
  print '  file_extension:        modify all files with this file extension in the current directory'
  print '                         example: python updated_table_entry.py .ups'
  print '  --help, -help, -h:     print this message '
  exit()

args = sys.argv

if len(args) != 2: usage()
if args[1] == '-h': usage()
if args[1] == '--help': usage()
if args[1] == '-help': usage()

for filename in os.listdir('.'):

  if filename.endswith(args[1]):

    print 'Fixing file: ', filename

    fix_ups(filename)



