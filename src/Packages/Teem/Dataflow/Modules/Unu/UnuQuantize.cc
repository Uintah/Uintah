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
 *  UnuQuantize
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

#include <sstream>
#include <iostream>
using std::endl;
#include <stdio.h>

namespace SCITeem {

class UnuQuantize : public Module {
  NrrdIPort* inrrd_;
  NrrdOPort* onrrd_;
  GuiDouble minf_;
  GuiDouble maxf_;
  GuiInt nbits_;
  double last_minf_;
  double last_maxf_;
  int last_nbits_;
  int last_generation_;
  NrrdDataHandle last_nrrdH_;
public:
  UnuQuantize(GuiContext *ctx);
  virtual ~UnuQuantize();
  virtual void execute();
};

} // End namespace SCITeem
using namespace SCITeem;
DECLARE_MAKER(UnuQuantize)

UnuQuantize::UnuQuantize(GuiContext *ctx)
  : Module("UnuQuantize", ctx, Filter, "Unu", "Teem"),
    minf_(ctx->subVar("minf")),
    maxf_(ctx->subVar("maxf")),
    nbits_(ctx->subVar("nbits")), last_minf_(0),
    last_maxf_(0), last_nbits_(0), last_generation_(-1), last_nrrdH_(0)
{
}

UnuQuantize::~UnuQuantize() {
}

void 
UnuQuantize::execute()
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

  if (last_generation_ != nrrdH->generation) {
    // set default values for min,max
    nrrdMinMaxCleverSet(nrrdH->nrrd);
    cout << "min and max: " << nrrdH->nrrd->min << " " << nrrdH->nrrd->max 
	 << endl;
    ostringstream str;
    str << id.c_str() << " update_min_max " << nrrdH->nrrd->min 
	<< " " << nrrdH->nrrd->max << endl;

    gui->execute(str.str());
    minf_.reset();
    maxf_.reset();
  }



  double minf=minf_.get();
  double maxf=maxf_.get();
  int nbits=nbits_.get();
  if (last_generation_ == nrrdH->generation &&
      last_minf_ == minf &&
      last_maxf_ == maxf &&
      last_nbits_ == nbits &&
      last_nrrdH_.get_rep()) {
    onrrd_->send(last_nrrdH_);
    return;
  }
  // must detach because we are about to modify the input nrrd.
  last_generation_ = nrrdH->generation;
  nrrdH.detach(); 

  Nrrd *nin = nrrdH->nrrd;
  nin->min = minf;
  nin->max = maxf;

  msgStream_ << "Quantizing -- min="<<minf<<
    " max="<<maxf<<" nbits="<<nbits<<endl;
  NrrdData *nrrd = scinew NrrdData;
  if (nrrdQuantize(nrrd->nrrd = nrrdNew(), nin, nbits)) {
    char *err = biffGetDone(NRRD);
    error(string("Trouble quantizing: ") + err);
    free(err);
    return;
  }
  // propogate sci added data
  nrrd->copy_sci_data(*nrrdH.get_rep());

  last_minf_ = minf;
  last_maxf_ = maxf;
  last_nbits_ = nbits;
  last_nrrdH_ = nrrd;
  onrrd_->send(last_nrrdH_);
}

