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
 *  UnuConvert: Convert between C types
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>

#include <iostream>
using std::endl;
#include <stdio.h>

namespace SCITeem {

class UnuConvert : public Module {
  NrrdIPort* inrrd_;
  NrrdOPort* onrrd_;
  GuiInt type_;
  int last_type_;
  int last_generation_;
  NrrdDataHandle last_nrrdH_;
public:
  UnuConvert(SCIRun::GuiContext *ctx);
  virtual ~UnuConvert();
  virtual void execute();
};

} //end namespace SCITeem

using namespace SCITeem;
DECLARE_MAKER(UnuConvert)

UnuConvert::UnuConvert(SCIRun::GuiContext *ctx)
  : Module("UnuConvert", ctx, Filter, "Unu", "Teem"), 
  type_(ctx->subVar("type")),
  last_type_(0), last_generation_(-1), last_nrrdH_(0)
{
}

UnuConvert::~UnuConvert() {
}

void 
UnuConvert::execute()
{
  NrrdDataHandle nrrdH;
  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("Nrrd");
  onrrd_ = (NrrdOPort *)get_oport("Nrrd");

  if (!inrrd_) {
    error("Unable to initialize iport 'Nrrd'.");
    return;
  }
  if (!onrrd_) {
    error("Unable to initialize oport 'Nrrd'.");
    return;
  }


  if (!inrrd_->get(nrrdH))
    return;
  if (!nrrdH.get_rep()) {
    error("Empty input Nrrd.");
    return;
  }

  int type=type_.get();
  if (last_generation_ == nrrdH->generation &&
      last_type_ == type &&
      last_nrrdH_.get_rep()) {
    onrrd_->send(last_nrrdH_);
    return;
  }

  last_generation_ = nrrdH->generation;
  last_type_ = type;

  Nrrd *nin = nrrdH->nrrd;
  Nrrd *nout = nrrdNew();
  msgStream_ << "New type is "<<type<<endl;

  if (nrrdConvert(nout, nin, type)) {
    char *err = biffGetDone(NRRD);
    error(string("Trouble resampling: ") + err);
    msgStream_ << "  input Nrrd: nin->dim="<<nin->dim<<"\n";
    free(err);
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;
  nrrd->copy_sci_data(*nrrdH.get_rep());
  last_nrrdH_ = nrrd;
  onrrd_->send(last_nrrdH_);
}

