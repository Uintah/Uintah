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
 *  NrrdConvert: Convert between C types
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

class NrrdConvert : public Module {
  NrrdIPort* inrrd_;
  NrrdOPort* onrrd_;
  GuiInt type_;
  int last_type_;
  int last_generation_;
  NrrdDataHandle last_nrrdH_;
public:
  NrrdConvert(const string& id);
  virtual ~NrrdConvert();
  virtual void execute();
};

extern "C" Module* make_NrrdConvert(const string& id)
{
    return new NrrdConvert(id);
}

NrrdConvert::NrrdConvert(const string& id)
  : Module("NrrdConvert", id, Filter, "Filters", "Teem"), type_("type", id, this),
    last_type_(0), last_generation_(-1), last_nrrdH_(0)
{
}

NrrdConvert::~NrrdConvert() {
}

void 
NrrdConvert::execute()
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
  cerr << "New type is "<<type<<endl;

  nrrdConvert(nout, nin, type);
  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;
  last_nrrdH_ = nrrd;
  onrrd_->send(last_nrrdH_);
}

} // End namespace SCITeem
