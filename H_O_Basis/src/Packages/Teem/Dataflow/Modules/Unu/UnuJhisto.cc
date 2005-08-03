/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

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
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class UnuJhisto : public Module {
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
  string          old_bins_;
  string          old_mins_;
  string          old_maxs_;
  string          old_type_;
  int             weight_generation_;
  int             inrrd1_generation_;
  int             num_inrrd2_;
  NrrdDataHandle  last_nrrdH_;


  unsigned int get_type(const string &t);
};


DECLARE_MAKER(UnuJhisto)
UnuJhisto::UnuJhisto(GuiContext* ctx)
  : Module("UnuJhisto", ctx, Source, "UnuAtoM", "Teem"),
    bins_(ctx->subVar("bins")), mins_(ctx->subVar("mins")), 
    maxs_(ctx->subVar("maxs")), type_(ctx->subVar("type")),
    old_bins_(""), old_mins_(""), old_maxs_(""), old_type_(""),
    weight_generation_(-1), inrrd1_generation_(-1),
    num_inrrd2_(-1)
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

  Nrrd *weight = 0;
  Nrrd *nin1 = 0;
  Nrrd *nout = nrrdNew();

  bool need_execute = false;

  if (!weight_->get(weight_handle)) {
    weight = 0;
    weight_generation_ = -1;
  }
  else {
    if (weight_handle->generation != weight_generation_) {
      need_execute = true;
      weight_generation_ = weight_handle->generation;
    }
    weight = weight_handle->nrrd;
  }

  if (!inrrd1_->get(nrrd_handle1))
    return;
  else
    nin1 = nrrd_handle1->nrrd;

  if (!nrrd_handle1.get_rep()) {
    error("Empty InputNrrd1.");
    return;
  }

  if (inrrd1_generation_ != nrrd_handle1->generation) {
    need_execute = true;
    inrrd1_generation_ = nrrd_handle1->generation;
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
    NrrdDataHandle nrrd;
    if (inrrd->get(nrrd) && nrrd.get_rep()) {
      if (nrrd->nrrd->dim > max_dim)
	max_dim = nrrd->nrrd->dim;
      
      nrrds.push_back(nrrd->nrrd);
    }
    ++pi; 
  }

  if ((nrrds.size()-1) != (unsigned)num_inrrd2_) {
    need_execute = true;
    num_inrrd2_ = nrrds.size()-1; // minus 1 accounts for inrrd1
  }

  reset_vars();

  if (old_bins_ != bins_.get() ||
      old_mins_ != mins_.get() ||
      old_maxs_ != maxs_.get() ||
      old_type_ != type_.get()) {
    old_bins_ = bins_.get();
    old_mins_ = mins_.get();
    old_maxs_ = maxs_.get();
    old_type_ = type_.get();
    need_execute = true;
  }


  if( need_execute) {
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
    double *min = new double[nrrds.size()];
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
    double *max = new double[nrrds.size()];
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
	    max[counter] = (atof(maxs.substr(start,end-start).c_str()));
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
	  max[counter] = (atof(maxs.substr(start,end-start).c_str()));
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

    // nrrdHistoJoint crashes if min == max
    for (int d = 0; d < (int)nrrds.size(); d++) {
      if (range[d]->min == range[d]->max) {
        warning("range has 0 width, not computing.");
        return;
      }
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


    NrrdData *nrrd = scinew NrrdData;
    nrrd->nrrd = nout;    
    last_nrrdH_ = nrrd;

    if (nrrds.size()) {

      if (airIsNaN(range[0]->min) || airIsNaN(range[0]->max)) {
	NrrdRange *minmax = nrrdRangeNewSet(nrrds[0], nrrdBlind8BitRangeFalse);
	if (airIsNaN(range[0]->min)) range[0]->min = minmax->min;
	if (airIsNaN(range[0]->max)) range[0]->max = minmax->max;
	nrrdRangeNix(minmax);
      }
      nrrdKeyValueAdd(last_nrrdH_->nrrd, "jhisto_nrrd0_min", 
		      to_string(range[0]->min).c_str());
      nrrdKeyValueAdd(last_nrrdH_->nrrd, "jhisto_nrrd0_max", 
		      to_string(range[0]->max).c_str());
    }
    
    for (int d=0; d < (int)nrrds.size(); ++d) {
      nrrdRangeNix(range[d]);
    }
    free(range);


    delete bin;
    delete min;
    delete max;
    
  }
  onrrd_->send(last_nrrdH_);
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


