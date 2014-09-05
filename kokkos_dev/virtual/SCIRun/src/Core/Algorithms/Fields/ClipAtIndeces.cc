//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : ClipAtIndeces.cc
//    Author : Martin Cole
//    Date   : Tue Aug  8 15:42:40 2006

#include <Core/Algorithms/Fields/ClipAtIndeces.h>

namespace SCIRun {

CompileInfoHandle
ClipAtIndecesBase::get_compile_info(const TypeDescription *fsrc)
  
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_name("ClipAtIndecesAlgo");
  static const string base_class_name("ClipAtIndecesBase");

  CompileInfo *rval = 
    scinew CompileInfo(template_name + "." +
		       fsrc->get_filename() + ".",
                       base_class_name, 
                       template_name, 
                       fsrc->get_name());

  rval->add_include(include_path);
  rval->add_basis_include("../src/Core/Basis/NoData.h");
  rval->add_basis_include("../src/Core/Basis/Constant.h");
  rval->add_basis_include("../src/Core/Basis/TriLinearLgn.h");
  rval->add_mesh_include("../src/Core/Datatypes/PointCloudMesh.h");
  rval->add_mesh_include("../src/Core/Datatypes/TriSurfMesh.h");
  fsrc->fill_compile_info(rval);

  return rval;
}

FieldHandle
clip_nodes(FieldHandle fh, const vector<unsigned int> &indeces)
{
  const TypeDescription *ftd = fh->get_type_description();
  // description for just the data in the field

  // Get the Algorithm.
  CompileInfoHandle ci = ClipAtIndecesBase::get_compile_info(ftd);

  DynamicAlgoHandle algo;
  if (!DynamicCompilation::compile(ci, algo)) {
    return 0;
  }
  
  ClipAtIndecesBase *clipper = (ClipAtIndecesBase*)algo.get_rep();
  if (! clipper) {
    cerr << "Error: could not get algorithm for ClipAtIndeces" 
	 << endl;
    return 0;
  }

  return clipper->clip_nodes(fh, indeces);
}

FieldHandle
clip_faces(FieldHandle fh, const vector<unsigned int> &indeces)
{
  const TypeDescription *ftd = fh->get_type_description();
  // description for just the data in the field

  // Get the Algorithm.
  CompileInfoHandle ci = ClipAtIndecesBase::get_compile_info(ftd);

  DynamicAlgoHandle algo;
  if (!DynamicCompilation::compile(ci, algo)) {
    return 0;
  }
  
  ClipAtIndecesBase *clipper = (ClipAtIndecesBase*)algo.get_rep();
  if (! clipper) {
    cerr << "Error: could not get algorithm for ClipAtIndeces" 
	 << endl;
    return 0;
  }

  return clipper->clip_faces(fh, indeces);
}

} // end namespace SCIRun
