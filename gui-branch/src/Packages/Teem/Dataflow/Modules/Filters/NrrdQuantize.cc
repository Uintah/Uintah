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
 *  NrrdQuantize
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
using std::cerr;
using std::endl;
#include <stdio.h>

namespace SCITeem {

class NrrdQuantize : public Module {
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
  NrrdQuantize(const string& id);
  virtual ~NrrdQuantize();
  virtual void execute();
};

extern "C" Module* make_NrrdQuantize(const string& id)
{
    return new NrrdQuantize(id);
}

NrrdQuantize::NrrdQuantize(const string& id)
  : Module("NrrdQuantize", id, Filter, "Filters", "Teem"), minf_("minf", id, this),
    maxf_("maxf", id, this), nbits_("nbits", id, this), last_minf_(0),
    last_maxf_(0), last_nbits_(0), last_generation_(-1), last_nrrdH_(0)
{
}

NrrdQuantize::~NrrdQuantize() {
}

void 
NrrdQuantize::execute()
{
  NrrdDataHandle nrrdH;
  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("Nrrd");
  onrrd_ = (NrrdOPort *)get_oport("Nrrd");

  if (!inrrd_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!onrrd_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  if (!inrrd_->get(nrrdH))
    return;
  if (!nrrdH.get_rep()) {
    error("Error: empty Nrrd\n");
    return;
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

  Nrrd *nin = nrrdH->nrrd;
  nin->min = minf;
  nin->max = maxf;

  cerr << "Quantizing -- min="<<minf<<" max="<<maxf<<" nbits="<<nbits<<endl;
  cerr << "nrrdMinMaxUse = "<<nrrdMinMaxUse<<"\n";
  NrrdData *nrrd = scinew NrrdData;
  if (nrrdQuantize(nrrd->nrrd = nrrdNew(), nin, nbits, nrrdMinMaxUse)) {
    char *err = biffGet(NRRD);
    fprintf(stderr, "NrrdQuantize: trouble quantizing:\n%s\n", err);
    free(err);
    return;
  }

  last_generation_ = nrrdH->generation;
  last_minf_ = minf;
  last_maxf_ = maxf;
  last_nbits_ = nbits;
  last_nrrdH_ = nrrd;
  onrrd_->send(last_nrrdH_);
}

} // End namespace SCITeem
