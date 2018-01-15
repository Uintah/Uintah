import sys
import os
from lxml import etree

"""
 Description:
   This script modifies all UPS files in the current directory from which you run the script
   with a user specified file extension. It updates the <MomentumSolver> node in the ups file
   to be compatible with the latest code by (if it currently is not) by adding a <wall_closure>
   node with a default model (constant coefficient).
   Note that your original ups file is stored with a *.orig_ups extension.

 Usage:
   python update_table_entry.py file_extension
   file_name:             modify the file with this name to update the table section in the UPS file.
                          example: python updated_table_entry.py myinput.ups
   --do_all_ups_files:    do all files in this directory with an .ups extension
   --do_all_xml_files:    do all files in this directory with an .xml extension
   --do_all_ups_ask_permission : do all files in this directory with an .ups extension but ask permission first per file.
   [--help, -help, -h]:   print this message
"""
def fix_ups(filename):

  parser = etree.XMLParser(remove_comments=False, remove_blank_text=True)
  tree = etree.parse(filename, parser)
  root = tree.getroot()

  CFD = root.find('CFD')
  Arches = CFD.find('ARCHES')
  ExplicitSolver = Arches.find('ExplicitSolver')
  MomentumSolver = ExplicitSolver.find('MomentumSolver')

  newEntry = etree.Element('wall_closure') 
  newEntry.attrib['type'] = 'constant_coefficient'
  MomentumSolver.insert(0,newEntry)

  ConvScheme = MomentumSolver.find("wall_closure")
  newEntry = etree.Element('wall_csmag')
  newEntry.text = '0.4'

  ConvScheme.insert(0,newEntry)

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
  print '  --do_all_ups_ask_permission : do all files in this directory with an .ups extension but ask permission first per file.'
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
elif args[1] == '--do_all_ups_ask_permission':
  for filename in os.listdir('.'):

    if filename.endswith('.ups'):

        print 'For file named: ', filename
        test = raw_input('Please indicate if you want it updated [y/n]: ')
        if test == 'y':
            fix_ups(filename)
        else:
            print 'Skiping this file. '


else:
    print 'Fixing file: ', args[1]
    fix_ups(args[1])

print 'Done. The original UPS file is saved with the extension *.orig_ups'
