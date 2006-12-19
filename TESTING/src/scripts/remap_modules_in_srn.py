#  
#  For more information, please see: http://software.sci.utah.edu
#  
#  The MIT License
#  
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
#  
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#  
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#  
#    File   : remap_modules_in_srn.py
#    Author : Martin Cole
#    Date   : Thu Oct 12 10:06:20 2006

from xml.dom.minidom import getDOMImplementation
from xml.dom import *
import sys
import re
import os

ws_exp = re.compile('\s')
pcm_exp = re.compile('(\w+)::(\w+)::(\w+)')
from_to_exp = re.compile('([\w:]+)\s+([\w:]+|###).*')
remap = {}

def get_document(fn) :
  impl = getDOMImplementation()
  doc = minidom.parse(fn)
  return doc

  
def swap_names(m) :
  global remap
  
  pkg = m.attributes['package'].nodeValue
  cat = m.attributes['category'].nodeValue
  nam = m.attributes['name'].nodeValue  

  key = "%s::%s::%s" % (pkg, cat, nam)
  if remap.has_key(key) :
    mo = pcm_exp.match(remap[key])
    m.setAttribute('package', mo.group(1))
    m.setAttribute('category', mo.group(2))
    m.setAttribute('name', mo.group(3))

def repair_file(fn) :
  doc = get_document(fn)
  walk_modules(doc.documentElement, swap_names)
  fd = open(fn, "w")
  doc.writexml(fd, "", "", '')
  doc.unlink()
  
def walk_modules(doc_node, action) :
  children = doc_node.childNodes
  for c in children :
    if c.nodeName == 'modules' :
      mc = c.childNodes
      for m in mc :
        if m.nodeName == 'module' :
          action(m)

def build_remap(fn) :
  global remap
  
  lines = []
  try:
    f = open(fn)
    lines = f.readlines()
  except:
    print "Failure reading %s" % fn
    return False
  
  lcount = 1
  for l in lines :
    mo = from_to_exp.match(l)
    if mo == None :
      print "line %d: Invalid input.\n\t(%s)" % (lcount, l[:-1])
    else :
      if mo.group(2) != '###' and mo.group(1) != mo.group(2) :
        remap[mo.group(1)] =  mo.group(2)
    lcount = lcount + 1
  if len(remap) :
    return True
  return False

svnp = re.compile("\.svn")
def repair_each(arg, dirname, names) :
  mo = svnp.search(dirname)
  if mo != None :
    return # do not work in .svn dirs!

  for f in names :
    fn = dirname + os.sep + f
    if os.path.isfile(fn) and fn[-4:] == '.srn' :
      repair_file(fn)

if __name__ == '__main__' :
  fn = ''
  remapping = ''
  try:
    fn = sys.argv[1]
    remapping = sys.argv[2]
  except:
    print "WARNING:   This script will overwrite the file or files specified"
    print "           Please backup your .srn files first."
    print "Usage:"
    print "python remap_modules_in_srn.py <path/to/.srn> <path/to/remapping_file>"
    print ''
    s = '\tIf the first argument is a directory, then work recursively \non all .srn files under that directory, otherwise the argument should\nspecify a single .srn file. The remapping_file contains the mapping\nfrom old to new module info.  The file should have one remapping per\nline like so: \n\noldpackage:::oldcategory::oldmodule  newpackage::newcategory::newmodule\n\n\tSee SCIRun/src/scripts/module-remapping.txt.\n\n'
    print s
    sys.exit(1)

  if build_remap(remapping) :

    try:
      if os.path.isdir(fn) :
        os.path.walk(fn, repair_each, ())
      else :
        repair_file(fn)
    except:
      print "Error: Bad input network file.  Is it really a .srn file?"
  else :
    print "Error: Bad or empty remapping file, exiting..."
