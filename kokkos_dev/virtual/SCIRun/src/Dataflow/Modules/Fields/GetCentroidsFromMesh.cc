/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

/*
 *  GetCentroidsFromMesh.cc:
 *
 *  Written by:
 *   moulding
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Modules/Fields/GetCentroidsFromMesh.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Core/Util/DynamicCompilation.h>
#include <math.h>

#include <vector>
#include <iostream>

namespace SCIRun {

using namespace std;

class GetCentroidsFromMesh : public Module {
public:
  GetCentroidsFromMesh(GuiContext* ctx);
  virtual ~GetCentroidsFromMesh();
  virtual void execute();
};


DECLARE_MAKER(GetCentroidsFromMesh)


GetCentroidsFromMesh::GetCentroidsFromMesh(GuiContext* ctx)
  : Module("GetCentroidsFromMesh", ctx, Filter, "NewField", "SCIRun")
{
}


GetCentroidsFromMesh::~GetCentroidsFromMesh()
{
}



void
GetCentroidsFromMesh::execute()
{
  FieldHandle ifieldhandle;
  if (!get_input_handle("TetVolField", ifieldhandle)) return;

  const TypeDescription *ftd = ifieldhandle->get_type_description();
  CompileInfoHandle ci = GetCentroidsFromMeshAlgo::get_compile_info(ftd);
  Handle<GetCentroidsFromMeshAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, this)) return;

  FieldHandle ofieldhandle(algo->execute(this, ifieldhandle));
  
  send_output_handle("PointCloudField", ofieldhandle);
}



CompileInfoHandle
GetCentroidsFromMeshAlgo::get_compile_info(const TypeDescription *field_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("GetCentroidsFromMeshAlgoT");
  static const string base_class_name("GetCentroidsFromMeshAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       field_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       field_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_basis_include("../src/Core/Basis/Constant.h");
  rval->add_mesh_include("../src/Core/Datatypes/PointCloudMesh.h");
  field_td->fill_compile_info(rval);
  return rval;
}

} // End namespace SCIRun


