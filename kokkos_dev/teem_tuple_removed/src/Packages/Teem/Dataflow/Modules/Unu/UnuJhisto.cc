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
 *  UnuJhisto.cc 
 *  the opposite of "crop".
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

class PSECORESHARE UnuJhisto : public Module {
public:
  UnuJhisto(GuiContext*);

  virtual ~UnuJhisto();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  NrrdIPort*      weight_;
  NrrdIPort*      inrrd1_;
  NrrdOPort*      onrrd_;

  GuiString       bins_;
  GuiString       mins_;
  GuiString       maxs_;
  GuiString       type_;

  unsigned int get_type(const string &t);
};


DECLARE_MAKER(UnuJhisto)
UnuJhisto::UnuJhisto(GuiContext* ctx)
  : Module("UnuJhisto", ctx, Source, "Unu", "Teem"),
    bins_(ctx->subVar("bins")), mins_(ctx->subVar("mins")), 
    maxs_(ctx->subVar("maxs")), type_(ctx->subVar("type"))    
{
}

UnuJhisto::~UnuJhisto(){
}

void
 UnuJhisto::execute(){
  NrrdDataHandle weight_handle;
  NrrdDataHandle nrrd_handle1;

  update_state(NeedData);
  weight_ = (NrrdIPort *)get_iport("WeightNrrd");
  inrrd1_ = (NrrdIPort *)get_iport("InputNrrd1");
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!weight_) {
    error("Unable to initialize iport 'WeightNrrd'.");
    return;
  }
  if (!inrrd1_) {
    error("Unable to initialize iport 'InputNrrd1'.");
    return;
  }
  if (!onrrd_) {
    error("Unable to initialize oport 'OutputNrrd'.");
    return;
  }

  Nrrd *weight = 0;
  Nrrd *nin1 = 0;
  Nrrd *nout = nrrdNew();

  if (!weight_->get(weight_handle))
    weight = 0;
  else 
    weight = weight_handle->nrrd;

  if (!inrrd1_->get(nrrd_handle1))
    return;
  else
    nin1 = nrrd_handle1->nrrd;

  if (!nrrd_handle1.get_rep()) {
    error("Empty InputNrrd1.");
    return;
  }

  port_range_type nrange = get_iports("InputNrrd2");
  if (nrange.first == nrange.second) { return; }

  vector<Nrrd*> nrrds;
  nrrds.push_back(nrrd_handle1->nrrd);

  int max_dim = nin1->dim;
  port_map_type::iterator pi = nrange.first;
  while (pi != nrange.second)
  {
    NrrdIPort *inrrd = (NrrdIPort *)get_iport(pi->second);
    if (!inrrd) {
      error("Unable to initialize iport '" + to_string(pi->second) + "'.");
      return;
    }

    NrrdDataHandle nrrd;
    if (inrrd->get(nrrd) && nrrd.get_rep()) {
      if (nrrd->nrrd->dim > max_dim)
	max_dim = nrrd->nrrd->dim;
      
      nrrds.push_back(nrrd->nrrd);
    }
    ++pi; 
  }

  reset_vars();

  // Determine the number of bins given
  string bins = bins_.get();
  int binsLen = 0;
  char ch;
  int i=0, start=0;
  bool inword = false;
  while (i < (int)bins.length()) {
    ch = bins[i];
    if(isspace(ch)) {
      if (inword) {
	binsLen++;
	inword = false;
      }
    } else if (i == (int)bins.length()-1) {
      binsLen++;
      inword = false;
    } else {
      if(!inword) 
	inword = true;
    }
    i++;
  }

  if ((int)nrrds.size() != binsLen) {
    error("Number of input nrrds is not equal to number of bin specifications.");
    return;
  }

  // get bins
  int *bin = new int[nrrds.size()];
  i=0, start=0;
  int which = 0, end=0, counter=0;
  inword = false;
  while (i < (int)bins.length()) {
    ch = bins[i];
    if(isspace(ch)) {
      if (inword) {
	end = i;
	bin[counter] = (atoi(bins.substr(start,end-start).c_str()));
	which++;
	counter++;
	inword = false;
      }
    } else if (i == (int)bins.length()-1) {
      if (!inword) {
	start = i;
      }
      end = i+1;
      bin[counter] = (atoi(bins.substr(start,end-start).c_str()));
      which++;
      counter++;
      inword = false;
    } else {
      if(!inword) {
	start = i;
	inword = true;
      }
    }
    i++;
  }

  NrrdRange **range = (NrrdRange **)calloc(nrrds.size(), sizeof(NrrdRange*));
  for (int d=0; d<(int)nrrds.size(); d++) {
    range[d] = nrrdRangeNew(AIR_NAN, AIR_NAN);
  }

  // Determine the number of mins given
  string mins = mins_.get();
  int minsLen = 0;
  i = 0;
  inword = false;
  while (i < (int)mins.length()) {
    ch = mins[i];
    if(isspace(ch)) {
      if (inword) {
	minsLen++;
	inword = false;
      }
    } else if (i == (int)mins.length()-1) {
      minsLen++;
      inword = false;
    } else {
      if(!inword) 
	inword = true;
    }
    i++;
  }

  if ((int)nrrds.size() != minsLen) {
    error("Number of input nrrds is not equal to number of mins specifications.");
    return;
  }

  // get mins
  int *min = new int[nrrds.size()];
  i=0, start=0;
  which = 0, end=0, counter=0;
  inword = false;
  while (i < (int)mins.length()) {
    ch = mins[i];
    if(isspace(ch)) {
      if (inword) {
	end = i;
	if (mins.substr(start,end-start) == "nan")
	  min[counter] = AIR_NAN;
	else
	  min[counter] = (atoi(mins.substr(start,end-start).c_str()));
	which++;
	counter++;
	inword = false;
      }
    } else if (i == (int)mins.length()-1) {
      if (!inword) {
	start = i;
      }
      end = i+1;
      if (mins.substr(start,end-start) == "nan")
	min[counter] = AIR_NAN;
      else
	min[counter] = (atoi(mins.substr(start,end-start).c_str()));
      which++;
      counter++;
      inword = false;
    } else {
      if(!inword) {
	start = i;
	inword = true;
      }
    }
    i++;
  }


  for (int d=0; d<(int)nrrds.size(); d++) {
    range[d]->min = min[d];
  }

  // Determaxe the number of maxs given
  string maxs = maxs_.get();
  int maxsLen = 0;
  inword = false;
  i = 0;
  while (i < (int)maxs.length()) {
    ch = maxs[i];
    if(isspace(ch)) {
      if (inword) {
	maxsLen++;
	inword = false;
      }
    } else if (i == (int)maxs.length()-1) {
      maxsLen++;
      inword = false;
    } else {
      if(!inword) 
	inword = true;
    }
    i++;
  }

  if ((int)nrrds.size() != maxsLen) {
    error("Number of input nrrds is not equal to number of maxs specifications.");
    return;
  }

  // get maxs
  int *max = new int[nrrds.size()];
  i=0, start=0;
  which = 0, end=0, counter=0;
  inword = false;
  while (i < (int)maxs.length()) {
    ch = maxs[i];
    if(isspace(ch)) {
      if (inword) {
	end = i;
	if (maxs.substr(start,end-start) == "nan")
	  max[counter] = AIR_NAN;
	else
	  max[counter] = (atoi(maxs.substr(start,end-start).c_str()));
	which++;
	counter++;
	inword = false;
      }
    } else if (i == (int)maxs.length()-1) {
      if (!inword) {
	start = i;
      }
      end = i+1;
      if (maxs.substr(start,end-start) == "nan")
	max[counter] = AIR_NAN;
      else
	max[counter] = (atoi(maxs.substr(start,end-start).c_str()));
      which++;
      counter++;
      inword = false;
    } else {
      if(!inword) {
	start = i;
	inword = true;
      }
    }
    i++;
  }

  for (int d=0; d<(int)nrrds.size(); d++) {
    range[d]->max = max[d];
  }

  int clamp[NRRD_DIM_MAX];
  for (int d=0; d<(int)nrrds.size(); d++) {
    clamp[d] = 0;
  }

  if (nrrdHistoJoint(nout, (const Nrrd**)&nrrds[0], 
		     (const NrrdRange**)range,
		     (int)nrrds.size(), weight, bin, get_type(type_.get()), 
		     clamp)) {
    char *err = biffGetDone(NRRD);
    error(string("Error performing Unu Jhisto: ") +  err);
    free(err);
    return;
  }
    
  // make call
  
  delete bin;
  delete min;
  delete max;


  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  onrrd_->send(out);
}

void
 UnuJhisto::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

unsigned int 
UnuJhisto::get_type(const string &type) {
  if (type == "nrrdTypeChar") 
    return nrrdTypeChar;
  else if (type == "nrrdTypeUChar")  
    return nrrdTypeUChar;
  else if (type == "nrrdTypeShort")  
    return nrrdTypeShort;
  else if (type == "nrrdTypeUShort") 
    return nrrdTypeUShort;
  else if (type == "nrrdTypeInt")  
    return nrrdTypeInt;
  else if (type == "nrrdTypeUInt")   
    return nrrdTypeUInt;
  else if (type == "nrrdTypeFloat") 
    return nrrdTypeFloat;
  else if (type == "nrrdTypeDouble")  
    return nrrdTypeDouble;
  else    
    return nrrdTypeUInt;
}

} // End namespace Teem


