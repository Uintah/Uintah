#! /usr/local/bin/python
#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#
# File:		field_dispatch_macro.py
# Author:	Martin Cole
# Created:	Thu Mar 15 10:43:32 MST 2001
# 
# Description:	create macro to callback templated function with completely
#               typed fields.

import os
from sys import argv

fields = ("LevelField",)
nonlattice = ()
data_types = ()
scalar = ("double", "long", "float")
non = ("Vector", "Matrix3")

#fields = ("TetVol", "LatticeVol")
#data_types = ("double", "int")

def indent(n) :
  s = ""
  i = 0
  while i < n :
    s = s + "  "
    i = i + 1
  return s
  
def recurse_macro(n, orig_num, ind, out) :
  
  for f in fields :
    last = len(out) - 1
    out[last] = out[last] + 'if (disp_name == "%s") {\\\n' % (f)
    ind = ind + 1
    out.append(indent(ind))
    for d in data_types :
      dlast = len(out) - 1
      out[dlast] = (out[dlast] +
                    'if (field%d->get_type_name(1) == "%s") {\\\n' % (n,d))

      ind = ind + 1
      out.append(indent(ind) + '%s<%s> *f%d = 0;\\\n' % (f, d, n))
      out.append(indent(ind) +
                 'f%d = dynamic_cast<%s<%s>*>(field%d.get_rep());\\\n' %
                 (n, f, d, n))
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
                 'disp_error = true; ' +
                 'disp_msg = "%s<%s>::get_type_name is broken";\\\n' % (f, d))
      ind = ind - 1
      out.append(indent(ind) +'}\\\n')
      ind = ind - 1
      if d == data_types[len(data_types) -1] :          
        out.append(indent(ind) + '}\\\n')
      else :
        out.append(indent(ind) + '} else ')

    ind = ind - 1
    if f == fields[len(fields) -1] : 
      out.append(indent(ind) + '} else if (disp_error) {\\\n')
      ind = ind + 1
      out.append(indent(ind) + 'cerr << "Error: " << disp_msg << endl;\\\n')
      ind = ind - 1
      out.append(indent(ind) + '}\\\n')
    else :
      out.append(indent(ind) + '} else ')

def write_macro(n, out, name) :
  i = 1
  s = "#define %s%d(" % (name, n)
  while i <= n :
    s = s + "field%d, " % i;
    i = i + 1
  s = s + "callback)\\\n"
  ind = 1
  out.append(s)
  out.append(indent(ind) + "bool disp_error = false;\\\n")
  out.append(indent(ind) + "string disp_msg;\\\n")
  out.append(indent(ind) +
             "string disp_name = field%d->get_type_name(0);\\\n" % n)
  out.append(indent(ind))
  recurse_macro(n, n, ind, out)

if __name__ == '__main__' :
  global data_types
  data_types = scalar + non
  
  num = int(argv[1])  
  print "num recursions requested %d" % num
  out = []
  i = 1
  while i <= num :
    fp = open("Dispatch%d.h" % i, "w")
    out.append("//  This is a generated file, DO NOT EDIT!!\n")
    out.append("\n")
    out.append("#if !defined(Datatypes_Dispatch%d_h)\n" % i)
    out.append("#define Datatypes_Dispatch%d_h\n" % i)
    out.append("\n")
    out.append("// dispatch%d macro follows\n" % i)
    write_macro(i, out, "uintah_dispatch")
    out.append("\n")
    out.append("#endif //Datatypes_Dispatch%d_h\n" % i)
    fp.writelines(out);
    out = []
    i = i + 1
    fp.close()

  i = 1
  data_types = scalar
  while i <= num :
    fp = open("DispatchScalar%d.h" % i, "w")
    out.append("//  This is a generated file, DO NOT EDIT!!\n")
    out.append("\n")
    out.append("#if !defined(Datatypes_Dispatch%d_h)\n" % i)
    out.append("#define Datatypes_Dispatch%d_h\n" % i)
    out.append("\n")
    out.append("// dispatch_scalar%d macro follows\n" % i)
    write_macro(i, out, "uintah_dispatch_scalar")
    out.append("\n")
    out.append("#endif //Datatypes_Dispatch%d_h\n" % i)
    fp.writelines(out);
    out = []
    i = i + 1
    fp.close()

  i = 1
  data_types = non
  while i <= num :
    fp = open("DispatchNonScalar%d.h" % i, "w")
    out.append("//  This is a generated file, DO NOT EDIT!!\n")
    out.append("\n")
    out.append("#if !defined(Datatypes_Dispatch%d_h)\n" % i)
    out.append("#define Datatypes_Dispatch%d_h\n" % i)
    out.append("\n")
    out.append("// dispatch_non_scalar%d macro follows\n" % i)
    write_macro(i, out, "uintah_dispatch_non_scalar")
    out.append("\n")
    out.append("#endif //Datatypes_Dispatch%d_h\n" % i)
    fp.writelines(out);
    out = []
    i = i + 1
    fp.close()
