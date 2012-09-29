#! /usr/local/bin/python

#
#  Copyright (c) 1997-2012 The University of Utah
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the \"Software\"), to
#  deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
#  sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#  IN THE SOFTWARE.
# 
# 
# 
# 
# 
# File:		cd_templates_fields.py
# Author:	Michael Callahan
# Created:	Sept 2001
# 
# Description:	create field templates


import os
from sys import argv



ufields = ["LatVolField",
           "TetVolField",
           "HexVolField",
           "TriSurfField",
           "ImageField",
           "CurveField",
           "ScanlineField",
           "PointCloudField"
           "QuadraticTetVolField"
           ]

mfields = ["MaskedTetVolField",
           "MaskedHexVolField",
           "MaskedLatVolField",
           "MaskedTriSurfField"
           ]

fields = ufields + mfields

meshes = [("LatVolMesh", "FData3d"),
          ("TetVolMesh", "vector"),
          ("HexVolMesh", "vector"),
          ("TriSurfMesh", "vector"),
          ("ImageMesh", "FData2d"),
          ("CurveMesh", "vector"),
          ("ScanlineMesh", "vector"),
          ("PointCloudMesh", "vector")
          ("QuadraticTetVolMesh", "vector")
          ]

containers = ["vector", "FData3d", "FData2d"]

tdatas = ["Tensor"]
vdatas = ["Vector"]
sdatas = ["double", "float", "int", "short", "unsigned char"]
odatas = []
datas = tdatas + vdatas + sdatas + odatas


if __name__ == '__main__' :

  out = open("cd_templates_fields.cc", "w")

  out.write("#include <Core/Persistent/PersistentSTL.h>\n")
  out.write("#include <Core/Geometry/Tensor.h>\n")
  out.write("#include <Core/Geometry/Vector.h>\n")
  out.write("#include <Core/Datatypes/GenericField.h>\n")
  for f in fields :
    out.write("#include <Core/Datatypes/" + f + ".h>\n")
  out.write("\n")
  out.write("\n")
  out.write("using namespace SCIRun;\n")
  out.write("\n")
  out.write("\n")

  for f in fields :
    for d in datas :
      out.write("template class " + f + "<" + d + ">;\n")
    out.write("\n")
    for d in datas :
      out.write("const TypeDescription* get_type_description("
                + f + "<" + d + "> *);\n")
    out.write("\n")
  out.write("\n")

  for f in ufields :
    for d in tdatas :
      out.write("template <>\n")
      out.write("TensorFieldInterface *\n")
      out.write(f + "<" + d + ">::query_tensor_interface() const\n")
      out.write("{\n")
      out.write("  return scinew TFInterface<" + f + "<" + d + "> >(this);\n")
      out.write("}\n")
      out.write("\n")
    out.write("\n")

    for d in vdatas :
      out.write("template <>\n")
      out.write("VectorFieldInterface *\n")
      out.write(f + "<" + d + ">::query_vector_interface() const\n")
      out.write("{\n")
      out.write("  return scinew VFInterface<" + f + "<" + d + "> >(this);\n")
      out.write("}\n")
      out.write("\n")
    out.write("\n")

    for d in sdatas :
      out.write("template <>\n")
      out.write("ScalarFieldInterface *\n")
      out.write(f + "<" + d + ">::query_scalar_interface() const\n")
      out.write("{\n")
      out.write("  return scinew SFInterface<" + f + "<" + d + "> >(this);\n")
      out.write("}\n")
      out.write("\n")
    out.write("\n")
  out.write("\n")

  for mc in meshes :
    for d in datas :
      out.write("template class GenericField<"
                 + mc[0] + ", " + mc[1] + "<" + d + "> >;\n")
    out.write("\n")
  out.write("\n")

  for c in containers :
    for d in datas :
      out.write("template class " + c + "<" + d + ">;\n")
    out.write("\n")
  out.write("\n")

  out.close();
