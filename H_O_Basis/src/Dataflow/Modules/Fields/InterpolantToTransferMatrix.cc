/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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
					  fitp_h->order_type_description());
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
