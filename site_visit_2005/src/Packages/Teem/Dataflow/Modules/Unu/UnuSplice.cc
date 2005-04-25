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
 *  UnuSplice.cc Replace a slice with a different nrrd. This is functionally the
 *  opposite of "slice".
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

#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class UnuSplice : public Module {
public:
  UnuSplice(GuiContext*);

  virtual ~UnuSplice();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  NrrdIPort*      inrrd_;
  NrrdIPort*      islice_;
  NrrdOPort*      onrrd_;

  GuiInt       axis_;
  GuiInt       position_;
};


DECLARE_MAKER(UnuSplice)
UnuSplice::UnuSplice(GuiContext* ctx)
  : Module("UnuSplice", ctx, Source, "UnuNtoZ", "Teem"),
    inrrd_(0), islice_(0), onrrd_(0),
    axis_(ctx->subVar("axis")),
    position_(ctx->subVar("position"))
{
}

UnuSplice::~UnuSplice(){
}

void
 UnuSplice::execute(){
  NrrdDataHandle nrrd_handle;
  NrrdDataHandle slice_handle;

  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("InputNrrd");
  islice_ = (NrrdIPort *)get_iport("SliceNrrd");
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inrrd_->get(nrrd_handle))
    return;
  if (!islice_->get(slice_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty InputNrrd.");
    return;
  }
  if (!slice_handle.get_rep()) {
    error("Empty SliceNrrd.");
    return;
  }

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *slice = slice_handle->nrrd;
  Nrrd *nout = nrrdNew();

  // position could be an integer or M-<integer>
  if (!( AIR_IN_CL(0, axis_.get(), nin->dim-1) )) {
    error("Axis " + to_string(axis_.get()) + " not in range [0," + to_string(nin->dim-1) + "]");
    return;
  }

  // FIX ME (ability to have M-<int>)
  
  if (nrrdSplice(nout, nin, slice, axis_.get(), position_.get())) {
    char *err = biffGetDone(NRRD);
    error(string("Error Splicing nrrd: ") + err);
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

  onrrd_->send(out);
}

void
 UnuSplice::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Teem


