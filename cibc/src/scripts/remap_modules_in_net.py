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
#    File   : remap_modules_in_net.py
#    Author : Martin Cole
#    Date   : Thu Oct 12 10:06:20 2006

import sys
import re
import os

ws_exp = re.compile('\s')
pcm_exp = re.compile('(\w+)::(\w+)::(\w+)')
from_to_exp = re.compile('([\w:]+)\s+([\w:]+|###).*')
remap = {}

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

mod_exp = re.compile('(.+addModuleAtPosition)\s+"(\w+)"\s+"(\w+)"\s+"(\w+)"(.+)')
def parse_net(fn) :
  global remap

  outfile = []
  f = open(fn, 'r')
  for ln in f.readlines() :
    mo = mod_exp.match(ln)
    if mo != None :
      key = "%s::%s::%s" % (mo.group(2), mo.group(3), mo.group(4))
      if remap.has_key(key) :
        nmo = pcm_exp.match(remap[key])
        newln =  '%s "%s" "%s" "%s"%s\n' % (mo.group(1), nmo.group(1),
                                            nmo.group(2), nmo.group(3),
                                            mo.group(5))
        outfile.append(newln)
      else :
        outfile.append(ln)
    else :
      outfile.append(ln)

  f = open(fn, 'w')
  f.writelines(outfile)


fld_exp = re.compile('(SCIRun)::(Fields)\w+::(\w+)')
        
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
        mo1 = fld_exp.match(mo.group(1))
        if mo1 != None :
          bwcompat = "%s::%s::%s" % (mo1.group(1), mo1.group(2), mo1.group(3))
          remap[bwcompat] = mo.group(2)
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
    if os.path.isfile(fn) and fn[-4:] == '.net' :
      parse_net(fn)

if __name__ == '__main__' :
  fn = ''
  remapping = ''
  try:
    fn = sys.argv[1]
    remapping = sys.argv[2]
  except:
    print "problem with arguments"
    print "python remap_modules_in_net.py <path/to/.srn> <path/to/remapping_file>"
    sys.exit(1)

  if build_remap(remapping) :

    if os.path.isdir(fn) :
      os.path.walk(fn, repair_each, ())
      sys.exit(1)
  
    parse_net(fn)
  
  else :
    print "Bad or empty remapping table, exiting..."
