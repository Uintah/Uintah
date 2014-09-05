/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  InterpolantToTransferMatrix.cc:
 *
 *  Convert an interpolant field into a sparse transfer matrix.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2003
 *
 *  Copyright (C) 2003 SCI Institute
 */

#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Modules/Fields/InterpolantToTransferMatrix.h>
#include <Core/Containers/Handle.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

using std::vector;
using std::pair;


class InterpolantToTransferMatrix : public Module {
public:
  InterpolantToTransferMatrix(GuiContext* ctx);
  virtual ~InterpolantToTransferMatrix();

  virtual void execute();
};

DECLARE_MAKER(InterpolantToTransferMatrix)

InterpolantToTransferMatrix::InterpolantToTransferMatrix(GuiContext* ctx)
  : Module("InterpolantToTransferMatrix", ctx, Filter, "FieldsOther", "SCIRun")
{
}


InterpolantToTransferMatrix::~InterpolantToTransferMatrix()
{
}



void
InterpolantToTransferMatrix::execute()
{
  FieldIPort *itp_port = (FieldIPort *)get_iport("Interpolant");
  FieldHandle fitp_h;

  if (!itp_port) {
    error("Unable to initialize iport 'Interpolant'.");
    return;
  }
  if (!(itp_port->get(fitp_h) && fitp_h.get_rep()))
  {
    error("Could not get a handle or representation.");
    return;
  }

  CompileInfoHandle ci =
    Interp2TransferAlgo::get_compile_info(fitp_h->get_type_description(),
					  fitp_h->data_at_type_description());
  Handle<Interp2TransferAlgo> algo;
  if (!module_dynamic_compile(ci, algo))
  {
    error("Unsupported input field type, probably not an interpolant field.");
    return;
  }

  MatrixOPort *omp = (MatrixOPort *)getOPort("Transfer");
  if (!omp) {
    error("Unable to initialize oport 'Transfer'.");
    return;
  }
    
  MatrixHandle omatrixhandle(algo->execute(this, fitp_h));

  if (omatrixhandle.get_rep())
  {
    omp->send(omatrixhandle);
  }
}



CompileInfoHandle
Interp2TransferAlgo::get_compile_info(const TypeDescription *fitp,
				      const TypeDescription *litp)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("Interp2TransferAlgoT");
  static const string base_class_name("Interp2TransferAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fitp->get_filename() + "." +
		       litp->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       fitp->get_name() + ", " +
                       litp->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fitp->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
