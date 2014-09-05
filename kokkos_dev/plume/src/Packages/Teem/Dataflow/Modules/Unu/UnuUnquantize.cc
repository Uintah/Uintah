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
 *  UnuUnquantize.cc: Recover floating point values from quantized data
 *
 *  Written by:
 *   Darby Van Uitert
 *   April 2004
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>


namespace SCITeem {

using namespace SCIRun;

class UnuUnquantize : public Module {
public:
  UnuUnquantize(GuiContext*);

  virtual ~UnuUnquantize();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiInt       min_;
  GuiInt       useinputmin_;
  GuiInt       max_;
  GuiInt       useinputmax_;
  GuiInt       double_;
};


DECLARE_MAKER(UnuUnquantize)
UnuUnquantize::UnuUnquantize(GuiContext* ctx)
  : Module("UnuUnquantize", ctx, Source, "UnuNtoZ", "Teem"),
    min_(ctx->subVar("min")),
    useinputmin_(ctx->subVar("useinputmin")),
    max_(ctx->subVar("max")),
    useinputmax_(ctx->subVar("useinputmax")),
    double_(ctx->subVar("double"))
{
}

UnuUnquantize::~UnuUnquantize(){
}

void
 UnuUnquantize::execute(){

  NrrdDataHandle nrrd_handle;
  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("InputNrrd");
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");
  
  if (!inrrd_->get(nrrd_handle))
    return;
  
  if (!nrrd_handle.get_rep()) {
    error("Empty input Nrrd.");
    return;
  }
  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *nout = nrrdNew();
  
  Nrrd* copy = nrrdNew();
  nrrdCopy(copy, nin);

  if (!useinputmin_.get())
    copy->oldMin = min_.get();
  else
    copy->oldMin = nin->oldMin;

  if (!useinputmax_.get())
    copy->oldMax = max_.get();
  else 
    copy->oldMax = nin->oldMax;
  
  if (double_.get()) {
    if (nrrdUnquantize(nout, copy, nrrdTypeDouble)) {
      char *err = biffGetDone(NRRD);
      error(string("Error Unquantizing nrrd: ") + err);
      free(err);
    }
  } else {
    if (nrrdUnquantize(nout, copy, nrrdTypeFloat)) {
      char *err = biffGetDone(NRRD);
      error(string("Error Unquantizing nrrd: ") + err);
      free(err);
    }    
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  // Copy the properties.
  out->copy_properties(nrrd_handle.get_rep());

  // Copy the axis kinds
  for (int i=0; i<nin->dim; i++) {
    nout->axis[i].kind = nin->axis[i].kind;
  }

  onrrd_->send(out);
}

void
 UnuUnquantize::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Teem


