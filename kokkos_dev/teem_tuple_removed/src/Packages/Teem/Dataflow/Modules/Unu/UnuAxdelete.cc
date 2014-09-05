//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : UnuAxdelete.cc Remove one or more singleton axes from a nrrd.
//    Author : Darby Van Uitert
//    Date   : April 2004

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>
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
  Module("UnuAxdelete", ctx, Filter, "Unu", "Teem"), 
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

  if (!inrrd_) {
    error("Unable to initialize iport 'InputNrrd'.");
    return;
  }
  if (!onrrd_) {
    error("Unable to initialize oport 'OutputNrrd'.");
    return;
  }
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
    *((PropertyManager *) out.get_rep()) =
      *((PropertyManager *) nrrd_handle.get_rep());
    
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
    *((PropertyManager *) out.get_rep()) =
      *((PropertyManager *) nrrd_handle.get_rep());
    
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


