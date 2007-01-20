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
 *  SetupBEMatrix.cc: 
 *
 *  Written by:
 *   Andrew Keely - Northeastern University
 *   Michael Callahan - Department of Computer Science - University of Utah
 *   July 2006
 *
 *   Copyright (C) 2006 SCI Group
 */


#include <Packages/BioPSE/Dataflow/Modules/Forward/BuildBEMSrcToInfTransferMatrix.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>

namespace BioPSE {

using namespace SCIRun;


class BuildBEMSrcToInfTransferMatrix : public Module {
public:
  BuildBEMSrcToInfTransferMatrix(GuiContext* ctx);
  virtual ~BuildBEMSrcToInfTransferMatrix();
  virtual void execute();
};


DECLARE_MAKER(BuildBEMSrcToInfTransferMatrix)


BuildBEMSrcToInfTransferMatrix::BuildBEMSrcToInfTransferMatrix(GuiContext *context):
  Module("BuildBEMSrcToInfTransferMatrix", context, Source, "Forward", "BioPSE")
{
}


BuildBEMSrcToInfTransferMatrix::~BuildBEMSrcToInfTransferMatrix()
{
}


void
BuildBEMSrcToInfTransferMatrix::execute()
{
  FieldHandle surface, dipoles;
  if (!get_input_handle("Surface", surface)) return;
  if (!get_input_handle("Dipoles", dipoles)) return;
 
  // TODO: Check for point cloud of vectors in dipoles!
 
  const TypeDescription *mtd = surface->mesh()->get_type_description();
  const TypeDescription *ltd = surface->order_type_description();
  CompileInfoHandle ci = BuildBEMSrcToInfTransferMatrixAlgo::get_compile_info(mtd, ltd);
  Handle<BuildBEMSrcToInfTransferMatrixAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, this)) return;

  MatrixHandle output(algo->execute(this, surface, dipoles));
  
  send_output_handle("Surface Potentials", output);
}

} // end namespace BioPSE


namespace SCIRun {

CompileInfoHandle
BuildBEMSrcToInfTransferMatrixAlgo::get_compile_info(const TypeDescription *mesh_td,
                                          const TypeDescription *loc_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("BuildBEMSrcToInfTransferMatrixAlgoT");
  static const string base_class_name("BuildBEMSrcToInfTransferMatrixAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       mesh_td->get_filename() + "." +
                       loc_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       mesh_td->get_name() + ", " + loc_td->get_name());

  // Add in the include path to compile this obj.
  rval->add_include(include_path);
  mesh_td->fill_compile_info(rval);
  loc_td->fill_compile_info(rval);
  return rval;
}


} // end namespace SCIRun
