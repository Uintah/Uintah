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
 *  CalculateLatVolGradientsAtNodes.cc:  Unfinished modules
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   June 2004
 *
 *  Copyright (C) 2004 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Containers/Handle.h>

#include <Dataflow/Modules/Fields/CalculateLatVolGradientsAtNodes.h>

namespace SCIRun {

class CalculateLatVolGradientsAtNodes : public Module
{
public:
  CalculateLatVolGradientsAtNodes(GuiContext* ctx);
  virtual ~CalculateLatVolGradientsAtNodes();

  virtual void execute();

protected:
  FieldHandle fieldout_;

  int fGeneration_;
};


DECLARE_MAKER(CalculateLatVolGradientsAtNodes)

CalculateLatVolGradientsAtNodes::CalculateLatVolGradientsAtNodes(GuiContext* ctx)
  : Module("CalculateLatVolGradientsAtNodes", ctx, Filter, "ChangeFieldData", "SCIRun"),
    fGeneration_(-1)
{
}

CalculateLatVolGradientsAtNodes::~CalculateLatVolGradientsAtNodes()
{
}

void
CalculateLatVolGradientsAtNodes::execute()
{
  FieldHandle fieldin;
  if (!get_input_handle("Input Field", fieldin)) return;

  if (!fieldin->query_scalar_interface(this).get_rep() )
  {
    error( "This module only works on fields of scalar data.");
    return;
  }

  if (fieldin->basis_order() != 1)
  {
    error("This module only works on fields containing data at nodes.");
    return;
  }
  
  if (fieldin->get_type_description(Field::MESH_TD_E)->get_name().find("LatVolMesh", 0) == 
      string::npos)
  {
    error("This module only works on fields with based on LatVolMesh.");
  }

  // If no data or a changed recalcute.
  if( !fieldout_.get_rep() || fGeneration_ != fieldin->generation )
  {
    fGeneration_ = fieldin->generation;

    const TypeDescription *ftd = fieldin->get_type_description();

    CompileInfoHandle ci = CalculateLatVolGradientsAtNodesAlgo::get_compile_info(ftd);
    Handle<CalculateLatVolGradientsAtNodesAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    fieldout_ = algo->execute(fieldin);
  }

  send_output_handle("Output Gradient", fieldout_, true);
}



CompileInfoHandle
CalculateLatVolGradientsAtNodesAlgo::get_compile_info(const TypeDescription *ftd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("CalculateLatVolGradientsAtNodesAlgoT");
  static const string base_class_name("CalculateLatVolGradientsAtNodesAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name());
  
  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_basis_include("../src/Core/Basis/HexTrilinearLgn.h");
  rval->add_mesh_include("../src/Core/Datatypes/LatVolMesh.h");
  ftd->fill_compile_info(rval);
  return rval;
}

} // End namespace SCIRun
