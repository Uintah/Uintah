#! /usr/local/bin/python
#
#  The MIT License
#
#  Copyright (c) 1997-2012 The University of Utah
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to
#  deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
#  sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#  IN THE SOFTWARE.
# 
# 

# File:		mesh_dispatch_macro.py
# Author:	Martin Cole
# Created:	Thu Mar 15 10:43:32 MST 2001
# 
# Description:	create macro to callback templated function with completely
#               typed mesh.

import os
from sys import argv

meshes = ("TetVolMesh", "LatVolMesh", "TriSurfMesh", "ContourMesh", "PointCloudMesh")

def indent(n) :
  s = ""
  i = 0
  while i < n :
    s = s + "  "
    i = i + 1
  return s
  
def recurse_macro(n, orig_num, ind, out) :
  
  for f in meshes :
    last = len(out) - 1
    out[last] = out[last] + 'if (mdisp_name == "%s") {\\\n' % (f)
    ind = ind + 1
    out.append(indent(ind) + '%s *f%d = 0;\\\n' % (f, n))
    out.append(indent(ind) +
               'f%d = dynamic_cast<%s*>(mesh%d.get_rep());\\\n' % (n, f, n))
    out.append(indent(ind) + 'if (f%d) {\\\n' % n)
    ind = ind + 1
    out.append(indent(ind))
    # if done recursing call the callback
    if (n == 1) :
      s = 'callback('
      i = 1
      while i <= orig_num :
        s = s + "f%d" % i;
        if i < orig_num :
          s = s + ", "
        i = i + 1
      s = s + ');\\\n'
      clast = len(out) - 1
      out[clast] = out[clast] + s
    else :
      recurse_macro(n-1, orig_num, ind, out)

    ind = ind - 1
    out.append(indent(ind) + '} else {\\\n')
    ind = ind + 1
    out.append(indent(ind) +
               'mdisp_error = true; ' +
               'mdisp_msg = "%s::get_type_name is broken";\\\n' % f)
    ind = ind - 1
    out.append(indent(ind) +'}\\\n')

    ind = ind - 1
    if f == meshes[len(meshes) -1] : 
      out.append(indent(ind) + '} else if (mdisp_error) {\\\n')
      ind = ind + 1
      out.append(indent(ind) + 'cerr << "Error: " << mdisp_msg << endl;\\\n')
      ind = ind - 1
      out.append(indent(ind) + '}\\\n')
    else :
      out.append(indent(ind) + '} else ')

def write_macro(n, out, name) :
  i = 1
  s = "#define %s%d(" % (name, n)
  while i <= n :
    s = s + "mesh%d, " % i;
    i = i + 1
  s = s + "callback)\\\n"
  ind = 1
  out.append(s)
  out.append(indent(ind) + "bool mdisp_error = false;\\\n")
  out.append(indent(ind) + "string mdisp_msg;\\\n")
  out.append(indent(ind) +
             "string mdisp_name = mesh%d->get_type_name(0);\\\n" % n)
  out.append(indent(ind))
  recurse_macro(n, n, ind, out)

if __name__ == '__main__' :
  
  num = int(argv[1])  
  print "num recursions requested %d" % num
  out = []
  i = 1
  while i <= num :
    fp = open("DispatchMesh%d.h" % i, "w")
    out.append("//  This is a generated file, DO NOT EDIT!!\n")
    out.append("\n")
    out.append("#if !defined(Datatypes_DispatchMesh%d_h)\n" % i)
    out.append("#define Datatypes_DispatchMesh%d_h\n" % i)
    out.append("\n")
    out.append("// dispatch_mesh%d macro follows\n" % i)
    write_macro(i, out, "dispatch_mesh")
    out.append("\n")
    out.append("#endif //Datatypes_DispatchMesh%d_h\n" % i)
    fp.writelines(out);
    out = []
    i = i + 1
    fp.close()

