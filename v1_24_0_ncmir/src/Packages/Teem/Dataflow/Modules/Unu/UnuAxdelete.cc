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

//    File   : UnuAxdelete.cc Remove one or more singleton axes from a nrrd.
//    Author : Darby Van Uitert
//    Date   : April 2004

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class UnuAxdelete : public Module {
public:
  UnuAxdelete(SCIRun::GuiContext *ctx);
  virtual ~UnuAxdelete();
  virtual void execute();

private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiInt          axis_;
};

DECLARE_MAKER(UnuAxdelete)

UnuAxdelete::UnuAxdelete(SCIRun::GuiContext *ctx) : 
  Module("UnuAxdelete", ctx, Filter, "UnuAtoM", "Teem"), 
  axis_(ctx->subVar("axis"))
{
}

UnuAxdelete::~UnuAxdelete() {
}

void 
UnuAxdelete::execute()
{

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

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *nout = nrrdNew();

  int axis = axis_.get();
  int a = axis;
  if (axis == -1) {
    Nrrd *ntmp = nrrdNew();
    if (nrrdCopy(nout, nin)) {
      char *err = biffGetDone(NRRD);
      error(string("Error copying axis: ") + err);
      free(err);
    }
    for (a=0;
	 a<nout->dim && nout->axis[a].size > 1;
	 a++);
    while (a<nout->dim) {
      if (nrrdAxesDelete(ntmp, nout, a)
	  || nrrdCopy(nout, ntmp)) {
	char *err = biffGetDone(NRRD);
	error(string("Error Copying deleting axis: ") + err);
	free(err);
      }
      for (a=0;
	   a<nout->dim && nout->axis[a].size > 1;
	   a++);
    }
    NrrdData *nrrd = scinew NrrdData;
    nrrd->nrrd = nout;
    
    NrrdDataHandle out(nrrd);
    
    // Copy the properties.
    out->copy_properties(nrrd_handle.get_rep());
    
    onrrd_->send(out);
  } else {

    if (nrrdAxesDelete(nout, nin, axis)) {
      char *err = biffGetDone(NRRD);
      error(string("Error Axdeleting nrrd: ") + err);
      free(err);
    }

    NrrdData *nrrd = scinew NrrdData;
    nrrd->nrrd = nout;
    
    NrrdDataHandle out(nrrd);
    
    // Copy the properties.
    out->copy_properties(nrrd_handle.get_rep());
    
    // set kind
    // Copy the axis kinds
    int offset = 0;

    for (int i=0; i<nin->dim; i++) {
      if (i == axis) {
	offset = 1;
      } else 
	nout->axis[i-offset].kind = nin->axis[i].kind;
    }

    onrrd_->send(out);
  }

}

} // End namespace SCITeem


