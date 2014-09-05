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
 *  NrrdPermute
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

class NrrdPermute : public Module {
  NrrdIPort* inrrd_;
  NrrdOPort* onrrd_;
  GuiInt axis0_;
  GuiInt axis1_;
  GuiInt axis2_;
  int last_axis0_;
  int last_axis1_;
  int last_axis2_;
  int last_generation_;
  NrrdDataHandle last_nrrdH_;
public:
  int valid_data(int* axes);
  NrrdPermute(GuiContext *ctx);
  virtual ~NrrdPermute();
  virtual void execute();
};

} // End namespace SCITeem

using namespace SCITeem;
DECLARE_MAKER(NrrdPermute)

  NrrdPermute::NrrdPermute(GuiContext *ctx)
  : Module("NrrdPermute", ctx, Filter, "Filters", "Teem"),
    axis0_(ctx->subVar("axis0")), axis1_(ctx->subVar("axis1")),
    axis2_(ctx->subVar("axis2")), last_axis0_(0), last_axis1_(0), 
    last_axis2_(0), last_generation_(-1), last_nrrdH_(0)
{
}

NrrdPermute::~NrrdPermute() {
}


// check to see that the axes 0,1,2 are the entries (in some order) 
// in the axes array
int NrrdPermute::valid_data(int* axes) {
  int exists[3];
  exists[0]=exists[1]=exists[2]=0;
  for (int a=0; a<3; a++) {
    if (axes[a]>=0 && axes[a]<=2 && !exists[a])
      exists[a]=1;
    else {
      error("Bad axis assignments!");
      return 0;
    }
  }
  return 1;
}

void 
NrrdPermute::execute()
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

  int axes[3];
  axes[0]=axis0_.get();
  axes[1]=axis1_.get();
  axes[2]=axis2_.get();
  if (last_generation_ == nrrdH->generation &&
      last_axis0_ == axes[0] &&
      last_axis1_ == axes[1] &&
      last_axis2_ == axes[2] &&
      last_nrrdH_.get_rep()) {
    onrrd_->send(last_nrrdH_);
    return;
  }

  if (!valid_data(axes)) return;

  last_generation_ = nrrdH->generation;
  last_axis0_ = axes[0];
  last_axis1_ = axes[1];
  last_axis2_ = axes[2];

  Nrrd *nin = nrrdH->nrrd;
  Nrrd *nout = nrrdNew();
  msgStream_ << "Permuting: 0->"<<axes[0]<<" 1->"<<
    axes[1]<<" 2->"<<axes[2]<<endl;

  nrrdPermuteAxes(nout, nin, axes);
  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;
  last_nrrdH_ = nrrd;
  onrrd_->send(last_nrrdH_);
}

