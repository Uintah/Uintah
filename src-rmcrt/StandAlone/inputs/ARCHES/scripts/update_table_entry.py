import sys
import os
from lxml import etree

"""
 Description:
   This script modifies all tables in the current directory from which you run the script
   with a user specified file extension. It updates the <Properties> node in the ups file
   to be compatible with the latest code by (if it currently is not) by adding a <table>
   node with a generic label (which you can change).
   Note that your original ups file is stored with a *.orig_ups extension.

 Usage:
   python update_table_entry.py file_extension
   file_name:             modify the file with this name to update the table section in the UPS file.
                          example: python updated_table_entry.py myinput.ups
   --do_all_ups_files:    do all files in this directory with an .ups extension
   --do_all_xml_files:    do all files in this directory with an .xml extension
   [--help, -help, -h]:   print this message
"""
def fix_ups(filename):

  parser = etree.XMLParser(remove_comments=False, remove_blank_text=True)
  tree = etree.parse(filename, parser)
  root = tree.getroot()

  CFD = root.find('CFD')
  Arches = CFD.find('ARCHES')
  Properties = Arches.find('Properties')

  for n in Properties:

    Properties.remove(n)
    mytype=''
    found_one = 0
    if ( n.tag == 'ConstantProps'):
      mytype='constant'
      found_one = 1
    elif ( n.tag == "ClassicTable"):
      mytype='classic'
      found_one = 1
    elif ( n.tag == "ColdFlow" ):
      mytype='coldflow'
      found_one = 1

    if ( found_one == 1 ):
      newTab = etree.Element('table')
      newTab.attrib['label'] = 'a_user_generated_label'
      newTab.attrib['type'] = mytype
      for m in n:
        newTab.insert(0,m)
      Properties.insert(0,newTab)
    else:
      Properties.insert(0,n)

  os.system('cp '+filename+' '+filename+'.orig_ups')

  tree.write(filename, pretty_print=True, encoding="ISO-8859-1")


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
  print '  file_name:             modify the file with this name to update the table section in the UPS file.'
  print '                         example: python updated_table_entry.py myinput.ups'
  print '  --do_all_ups_files:    do all files in this directory with an .ups extension'
  print '  --do_all_xml_files:    do all files in this directory with an .xml extension'
  print '  [--help, -help, -h]:   print this message '
  exit()

args = sys.argv

if len(args) != 2: usage()
if args[1] == '-h': usage()
if args[1] == '--help': usage()
if args[1] == '-help': usage()


if args[1] == '--do_all_ups_files':
  for filename in os.listdir('.'):

    if filename.endswith('.ups'):

      print 'Fixing file: ', filename

      fix_ups(filename)
elif args[1] == '--do_all_xml_files':
  for filename in os.listdir('.'):

    if filename.endswith('.xml'):

      print 'Fixing file: ', filename

      fix_ups(filename)
else:
    print 'Fixing file: ', args[1]
    fix_ups(args[1])

print 'Done. The original UPS file is saved with the extension *.orig_ups'
