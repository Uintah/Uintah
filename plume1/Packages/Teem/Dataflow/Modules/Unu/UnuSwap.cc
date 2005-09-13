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

//    File   : UnuSwap.cc Interchange scan-line ordering of two axes
//    Author : Darby Van Uitert
//    Date   : April 2004

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>


namespace SCITeem {

using namespace SCIRun;

class UnuSwap : public Module {
public:
  UnuSwap(GuiContext*);

  virtual ~UnuSwap();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiInt          axisA_;
  GuiInt          axisB_;
};


DECLARE_MAKER(UnuSwap)
UnuSwap::UnuSwap(GuiContext* ctx)
  : Module("UnuSwap", ctx, Source, "UnuNtoZ", "Teem"),
    inrrd_(0), onrrd_(0), axisA_(ctx->subVar("axisA")),
    axisB_(ctx->subVar("axisB"))
{
}

UnuSwap::~UnuSwap(){
}

void
 UnuSwap::execute(){
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

  if (nrrdAxesSwap(nout, nin, axisA_.get(), axisB_.get())) {
    char *err = biffGetDone(NRRD);
    error(string("Error Swapping nrrd: ") + err);
    free(err);
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  // Copy the properties.
  out->copy_properties(nrrd_handle.get_rep());

  // Copy the axis kinds
  for (int i=0; i<nin->dim && i<nout->dim; i++)
  {
    nout->axis[i].kind = nin->axis[i].kind;
  }
  nout->axis[axisA_.get()].kind = nin->axis[axisB_.get()].kind;
  nout->axis[axisB_.get()].kind = nin->axis[axisA_.get()].kind;

  onrrd_->send(out);
}

void
 UnuSwap::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Teem


