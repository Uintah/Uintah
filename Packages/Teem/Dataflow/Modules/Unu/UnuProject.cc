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
//    File   : UnuProject.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>

#include <sstream>
#include <iostream>
using std::endl;
#include <stdio.h>

namespace SCITeem {

using namespace SCIRun;

class UnuProject : public Module {
public:
  UnuProject(SCIRun::GuiContext *ctx);
  virtual ~UnuProject();
  virtual void execute();

private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiInt       axis_;
  GuiInt    measure_;
};

DECLARE_MAKER(UnuProject)

UnuProject::UnuProject(SCIRun::GuiContext *ctx) : 
  Module("UnuProject", ctx, Filter, "Unu", "Teem"), 
  axis_(ctx->subVar("axis")),
  measure_(ctx->subVar("measure"))
{
}

UnuProject::~UnuProject() {
}

void 
UnuProject::execute()
{
  NrrdDataHandle nrrd_handle;
  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("nin");
  onrrd_ = (NrrdOPort *)get_oport("nout");

  if (!inrrd_) {
    error("Unable to initialize iport 'Nrrd'.");
    return;
  }
  if (!onrrd_) {
    error("Unable to initialize oport 'Nrrd'.");
    return;
  }
  if (!inrrd_->get(nrrd_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty input Nrrd.");
    return;
  }

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *nout = nrrdNew();

//  cerr << "axis_="<<axis_.get()<<"  measure_="<<measure_.get()<<"\n";

  if (nrrdProject(nout, nin, axis_.get(), measure_.get(), nrrdTypeDefault)) {
    char *err = biffGetDone(NRRD);
    error(string("Error Projecting nrrd: ") + err);
    free(err);
  }

  if (axis_.get() == 0) {  // fix tuple axis
    Nrrd *ntmp = nrrdNew();
    if (nrrdAxesInsert(ntmp, nout, 0)) {
      char *err = biffGetDone(NRRD);
      error(string("Error Inserting tuple axis: ") + err);
      free(err);
    }
    nrrdNuke(nout);
    nout = ntmp;
    delete nout->axis[0].label;
    nout->axis[0].label = strdup("Variance:Scalar");
  }
  
  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;
  nrrd->copy_sci_data(*nrrd_handle.get_rep());

  onrrd_->send(NrrdDataHandle(nrrd));
}

} // End namespace SCITeem
