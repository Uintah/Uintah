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
 *  NrrdCrop
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

class NrrdCrop : public Module {
  NrrdIPort* inrrd_;
  NrrdOPort* onrrd_;
  GuiString minAxis0_;
  GuiString maxAxis0_;
  GuiString minAxis1_;
  GuiString maxAxis1_;
  GuiString minAxis2_;
  GuiString maxAxis2_;
  string last_minAxis0_;
  string last_maxAxis0_;
  string last_minAxis1_;
  string last_maxAxis1_;
  string last_minAxis2_;
  string last_maxAxis2_;
  int last_generation_;
  NrrdDataHandle last_nrrdH_;
public:
  int valid_data(string *minS, string *maxS, int *min, int *max, Nrrd *nrrd);
  int getint(const char *str, int *n);
  NrrdCrop(SCIRun::GuiContext *ctx);
  virtual ~NrrdCrop();
  virtual void execute();
};

} // End namespace SCITeem
using namespace SCITeem;
DECLARE_MAKER(NrrdCrop)

NrrdCrop::NrrdCrop(SCIRun::GuiContext *ctx)
  : Module("NrrdCrop", ctx, Filter, "Filters", "Teem"), 
  minAxis0_(ctx->subVar("minAxis0")),
  maxAxis0_(ctx->subVar("maxAxis0")),
  minAxis1_(ctx->subVar("minAxis1")),
  maxAxis1_(ctx->subVar("maxAxis1")),
  minAxis2_(ctx->subVar("minAxis2")),
  maxAxis2_(ctx->subVar("maxAxis2")), 
  last_minAxis0_(std::string("")),
  last_maxAxis0_(std::string("")),
  last_minAxis1_(std::string("")),
  last_maxAxis1_(std::string("")),
  last_minAxis2_(std::string("")), 
  last_maxAxis2_(std::string("")),
  last_generation_(-1), last_nrrdH_(0)
{
}

NrrdCrop::~NrrdCrop() {
}

// edited from the Teem package: src/unrrdu/crop.c
// we initialize n with the axis size - 1
int
NrrdCrop::getint(const char *str, int *n) {
  if (!strlen(str)) return 1;
  if ('M' == str[0]) {
    if (1 < strlen(str)) {
      if (('+' == str[1] || '-' == str[1])) {
	int offset;
        if (1 != sscanf(str+1, "%d", &offset)) {
          return 1;
        }
	if (str[1] == '+') *n += offset;
	else *n -= offset;
        /* else we succesfully parsed the offset */
      }
      else {
        /* something other than '+' or '-' after 'M' */
        return 1;
      }
    }
  }
  else {
    if (1 != sscanf(str, "%d", n)) {
      return 1;
    }
    /* else we successfully parsed n */
  }
  return 0;
}

// look for M[+/-a] strings, as well just numbers
// set the actual indices in the min/max arrays
int NrrdCrop::valid_data(string *minS, string *maxS, 
			      int *min, int *max, Nrrd *nrrd) {
  string errstr("Error in NrrdCrop - bad subvolume range.");
  for (int a=0; a<3; a++) {
    const char *mins = minS[a].c_str();
    const char *maxs = maxS[a].c_str();
    min[a]=max[a]=nrrd->axis[a].size-1;
    if (getint(mins, &(min[a]))) { msgStream_ << errstr; return 0; }
    if (getint(maxs, &(max[a]))) { msgStream_ << errstr; return 0; }
    if (min[a] >= max[a]) { msgStream_ << errstr; return 0; }
  }
  return 1;
}

void 
NrrdCrop::execute()
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
  
  string minS[3], maxS[3];
  minS[0]=minAxis0_.get();
  maxS[0]=maxAxis0_.get();
  minS[1]=minAxis1_.get();
  maxS[1]=maxAxis1_.get();
  minS[2]=minAxis2_.get();
  maxS[2]=maxAxis2_.get();
  if (last_generation_ == nrrdH->generation &&
      last_minAxis0_ == minS[0] &&
      last_maxAxis0_ == maxS[0] &&
      last_minAxis1_ == minS[1] &&
      last_maxAxis1_ == maxS[1] &&
      last_minAxis2_ == minS[2] &&
      last_maxAxis2_ == maxS[2] &&
      last_nrrdH_.get_rep()) {
    onrrd_->send(last_nrrdH_);
    return;
  }

  int min[3], max[3];
  if (!valid_data(minS, maxS, min, max, nrrdH->nrrd)) return;

  last_minAxis0_ = minS[0];
  last_maxAxis0_ = maxS[0];
  last_minAxis1_ = minS[1];
  last_maxAxis1_ = maxS[1];
  last_minAxis2_ = minS[2];
  last_maxAxis2_ = maxS[2];
  last_generation_ = nrrdH->generation;

  Nrrd *nin = nrrdH->nrrd;
  Nrrd *nout = nrrdNew();
  msgStream_ << "Crop: ("<<min[0]<<","<<min[1]<<","<<min[2]<<") -> (";
  msgStream_ << max[0]<<","<<max[1]<<","<<max[2]<<")"<<endl;

  if (nrrdCrop(nout, nin, min, max)) {
    char *err = biffGetDone(NRRD);
    error(string("Trouble resampling: ") + err);
    msgStream_ << "  input Nrrd: nin->dim="<<nin->dim<<"\n";
    free(err);
  }
  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;
  last_nrrdH_ = nrrd;
  onrrd_->send(last_nrrdH_);
}

