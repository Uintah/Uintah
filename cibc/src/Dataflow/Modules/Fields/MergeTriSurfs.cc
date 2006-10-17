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
 *  MergeTriSurfs.cc: Take in fields and append them into one field.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   July 2004
 *
 *  Copyright (C) 2004 SCI Group
 */
#include <Core/Basis/Constant.h>
#include <Core/Basis/NoData.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Dataflow/Modules/Fields/MergeTriSurfs.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Handle.h>
#include <iostream>

namespace SCIRun {

class MergeTriSurfs : public Module {
public:
  MergeTriSurfs(GuiContext* ctx);
  virtual ~MergeTriSurfs();
  virtual void execute();
};

DECLARE_MAKER(MergeTriSurfs)
MergeTriSurfs::MergeTriSurfs(GuiContext* ctx)
  : Module("MergeTriSurfs", ctx, Filter, "NewField", "SCIRun")
{
}


MergeTriSurfs::~MergeTriSurfs()
{
}


void
MergeTriSurfs::execute()
{
  // Get input field.
  FieldHandle ifieldhandle;
  if (!get_input_handle("Input Field", ifieldhandle)) return;

  // TODO: Verify that it's a trisurf that we're testing.

  const TypeDescription *ftd = ifieldhandle->get_type_description();
  CompileInfoHandle ci = MergeTriSurfsAlgo::get_compile_info(ftd);
  Handle<MergeTriSurfsAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, this)) return;

  ifieldhandle.detach();
  ifieldhandle->mesh_detach();
  vector<unsigned int> new_nodes;
  vector<unsigned int> new_elems;
  algo->execute(this, ifieldhandle, new_nodes, new_elems);
  
  send_output_handle("Output Field", ifieldhandle, true);
}


CompileInfoHandle
MergeTriSurfsAlgo::get_compile_info(const TypeDescription *field_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("MergeTriSurfsAlgoT");
  static const string base_class_name("MergeTriSurfsAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       field_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       field_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  field_td->fill_compile_info(rval);
  return rval;
}

} // End namespace SCIRun
