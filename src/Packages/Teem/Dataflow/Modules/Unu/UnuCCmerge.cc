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
/*
 *  UnuCCmerge.cc 
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
#include <Teem/Dataflow/Ports/NrrdPort.h>

#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class PSECORESHARE UnuCCmerge : public Module {
public:
  UnuCCmerge(GuiContext*);

  virtual ~UnuCCmerge();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiInt       dir_;
  GuiInt       maxsize_;
  GuiInt       maxneigh_;
  GuiInt       connectivity_;
};


DECLARE_MAKER(UnuCCmerge)
UnuCCmerge::UnuCCmerge(GuiContext* ctx)
  : Module("UnuCCmerge", ctx, Source, "Unu", "Teem"),
    dir_(ctx->subVar("dir")),
    maxsize_(ctx->subVar("maxsize")),
    maxneigh_(ctx->subVar("maxneigh")),
    connectivity_(ctx->subVar("connectivity"))
{
}

UnuCCmerge::~UnuCCmerge(){
}

void
 UnuCCmerge::execute(){
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
    error("Empty InputNrrd.");
    return;
  }

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *nout = nrrdNew();

  if (nrrdCCMerge(nout, nin, 0, dir_.get(), maxsize_.get(), maxneigh_.get(), 
		  connectivity_.get())) {
    char *err = biffGetDone(NRRD);
    error(string("Error perfomring CC merge nrrd: ") + err);
    free(err);
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  onrrd_->send(out);
}

void
 UnuCCmerge::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Teem


