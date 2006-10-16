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
//    File   : BuildSurfNormals.cc
//    Author : Martin Cole
//    Date   : Mon Feb 27 10:39:28 2006


#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Modules/Fields/BuildSurfNormals.h>
#include <Core/Algorithms/Fields/ApplyMappingMatrix.h>

#include <iostream>

namespace SCIRun {

//! Module to build a surface field from a volume field.
class BuildSurfNormals : public Module {
public:
  BuildSurfNormals(GuiContext* ctx);
  virtual ~BuildSurfNormals();
  virtual void execute();

private:
  
  //! Input should be a volume field.
  FieldIPort*               infield_;
  int                       inmesh_gen_;

  //! Handle on the generated surface.
  MatrixHandle              normals_h_;
};

DECLARE_MAKER(BuildSurfNormals)
BuildSurfNormals::BuildSurfNormals(GuiContext* ctx) : 
  Module("BuildSurfNormals", ctx, Filter, "FieldsOther", "SCIRun"),
  inmesh_gen_(-1),
  normals_h_(0)
{
}

BuildSurfNormals::~BuildSurfNormals()
{
}


void 
BuildSurfNormals::execute()
{
  FieldHandle input;
  if (!get_input_handle("Surface Field", input)) return;

  // This module only can operate on surface fields.
  if (input->mesh()->dimensionality() != 2) 
  {
    error("Input field must be a Surface field.");
    return;
  }

  MeshHandle mesh = input->mesh();
  if (inmesh_gen_ != mesh->generation ||
      normals_h_.get_rep() == 0)
  {
    inmesh_gen_ = mesh->generation;

    const TypeDescription *mtd = mesh->get_type_description();
    CompileInfoHandle ci = BuildSurfNormalsAlgo::get_compile_info(mtd);
    Handle<BuildSurfNormalsAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) {
      error("Faild to get BuildSurfNormalsAlgo");
      return;
    }

    normals_h_ = algo->execute(this, mesh);
  }

  // Keep it around unless user selects port caching.
  send_output_handle("Nodal Surface Normals", normals_h_, true);
}


CompileInfoHandle
BuildSurfNormalsAlgo::get_compile_info(const TypeDescription *mesh_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("BuildSurfNormalsAlgoT");
  static const string base_class_name("BuildSurfNormalsAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       mesh_td->get_name(".", ".") + ".",
                       base_class_name, 
                       template_class_name, 
                       mesh_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  mesh_td->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun


