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
