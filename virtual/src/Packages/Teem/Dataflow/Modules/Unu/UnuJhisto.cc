/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class UnuJhisto : public Module {
public:
  UnuJhisto(GuiContext*);

  virtual ~UnuJhisto();

  virtual void execute();

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
};


DECLARE_MAKER(UnuJhisto)
UnuJhisto::UnuJhisto(GuiContext* ctx)
  : Module("UnuJhisto", ctx, Source, "UnuAtoM", "Teem"),
    bins_(get_ctx()->subVar("bins")), mins_(get_ctx()->subVar("mins")), 
    maxs_(get_ctx()->subVar("maxs")), type_(get_ctx()->subVar("type")),
    old_bins_(""), old_mins_(""), old_maxs_(""), old_type_(""),
    weight_generation_(-1), inrrd1_generation_(-1),
    num_inrrd2_(-1)
{
}


UnuJhisto::~UnuJhisto()
{
}


void
UnuJhisto::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrd_handle1;
  if (!get_input_handle("InputNrrd1", nrrd_handle1)) return;

  NrrdDataHandle weight_handle;
  get_input_handle("WeightNrrd", weight_handle, false);

  Nrrd *nin1 = nrrd_handle1->nrrd_;

  bool need_execute = false;

  Nrrd *weight = 0;
  if (!weight_handle.get_rep())
  {
    weight = 0;
    weight_generation_ = -1;
  }
  else
  {
    if (weight_handle->generation != weight_generation_)
    {
      need_execute = true;
      weight_generation_ = weight_handle->generation;
    }
    weight = weight_handle->nrrd_;
  }

  if (inrrd1_generation_ != nrrd_handle1->generation)
  {
    need_execute = true;
    inrrd1_generation_ = nrrd_handle1->generation;
  }

  port_range_type nrange = get_iports("InputNrrd2");
  if (nrange.first == nrange.second) { return; }

  vector<NrrdDataHandle> nrrds;
  nrrds.push_back(nrrd_handle1);

  int max_dim = nin1->dim;
  port_map_type::iterator pi = nrange.first;
  while (pi != nrange.second)
  {
    NrrdIPort *inrrd = (NrrdIPort *)get_iport(pi->second);
    NrrdDataHandle nrrd;
    if (inrrd->get(nrrd) && nrrd.get_rep())
    {
      if ((int)nrrd->nrrd_->dim > max_dim)
	max_dim = nrrd->nrrd_->dim;
      
      nrrds.push_back(nrrd);
    }
    ++pi; 
  }

  if ((nrrds.size()-1) != (unsigned)num_inrrd2_)
  {
    need_execute = true;
    num_inrrd2_ = nrrds.size()-1; // minus 1 accounts for inrrd1
  }

  reset_vars();

  if (old_bins_ != bins_.get() ||
      old_mins_ != mins_.get() ||
      old_maxs_ != maxs_.get() ||
      old_type_ != type_.get())
  {
    old_bins_ = bins_.get();
    old_mins_ = mins_.get();
    old_maxs_ = maxs_.get();
    old_type_ = type_.get();
    need_execute = true;
  }


  if( need_execute || !last_nrrdH_.get_rep())
  {
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
    size_t *bin = new size_t[nrrds.size()];
    i=0, start=0;
    int which = 0, end=0, counter=0;
    inword = false;
    while (i < (int)bins.length())
    {
      ch = bins[i];
      if(isspace(ch))
      {
	if (inword)
        {
	  end = i;
	  bin[counter] = (atoi(bins.substr(start,end-start).c_str()));
	  which++;
	  counter++;
	  inword = false;
	}
      }
      else if (i == (int)bins.length()-1)
      {
	if (!inword)
        {
	  start = i;
	}
	end = i+1;
	bin[counter] = (atoi(bins.substr(start,end-start).c_str()));
	which++;
	counter++;
	inword = false;
      }
      else
      {
	if(!inword)
        {
	  start = i;
	  inword = true;
	}
      }
      i++;
    }

    Nrrd **nrrds_array = scinew Nrrd *[nrrds.size()];
    NrrdRange **range = scinew NrrdRange *[nrrds.size()];
    for (unsigned int d = 0; d< nrrds.size(); d++)
    {
      nrrds_array[d] = nrrds[d]->nrrd_;
      range[d] = nrrdRangeNew(AIR_NAN, AIR_NAN);
    }
    
    // Determine the number of mins given
    string mins = mins_.get();
    int minsLen = 0;
    i = 0;
    inword = false;
    while (i < (int)mins.length())
    {
      ch = mins[i];
      if(isspace(ch))
      {
	if (inword)
        {
	  minsLen++;
	  inword = false;
	}
      }
      else if (i == (int)mins.length()-1)
      {
	minsLen++;
	inword = false;
      }
      else
      {
	if (!inword) 
	  inword = true;
      }
      i++;
    }
    
    if ((int)nrrds.size() != minsLen)
    {
      error("Number of input nrrds is not equal to number of mins specifications.");
      return;
    }
    
    // get mins
    double *min = new double[nrrds.size()];
    i=0, start=0;
    which = 0, end=0, counter=0;
    inword = false;
    while (i < (int)mins.length())
    {
      ch = mins[i];
      if(isspace(ch))
      {
	if (inword)
        {
	  end = i;
	  if (mins.substr(start,end-start) == "nan")
	    min[counter] = AIR_NAN;
	  else
	    min[counter] = (atoi(mins.substr(start,end-start).c_str()));
	  which++;
	  counter++;
	  inword = false;
	}
      }
      else if (i == (int)mins.length()-1)
      {
	if (!inword)
        {
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
      }
      else
      {
	if (!inword)
        {
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
    while (i < (int)maxs.length())
    {
      ch = maxs[i];
      if(isspace(ch))
      {
	if (inword)
        {
	  maxsLen++;
	  inword = false;
	}
      }
      else if (i == (int)maxs.length()-1)
      {
	maxsLen++;
	inword = false;
      }
      else
      {
	if(!inword) 
	  inword = true;
      }
      i++;
    }
    
    if ((int)nrrds.size() != maxsLen)
    {
      error("Number of input nrrds is not equal to number of maxs specifications.");
      return;
    }
    
    // get maxs
    double *max = new double[nrrds.size()];
    i=0, start=0;
    which = 0, end=0, counter=0;
    inword = false;
    while (i < (int)maxs.length())
    {
      ch = maxs[i];
      if(isspace(ch))
      {
	if (inword)
        {
	  end = i;
	  if (maxs.substr(start,end-start) == "nan")
	    max[counter] = AIR_NAN;
	  else
	    max[counter] = (atof(maxs.substr(start,end-start).c_str()));
	  which++;
	  counter++;
	  inword = false;
	}
      }
      else if (i == (int)maxs.length()-1)
      {
	if (!inword)
        {
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
      }
      else
      {
	if(!inword)
        {
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

    Nrrd *nout = nrrdNew();

    if (nrrdHistoJoint(nout, nrrds_array, range,
		       (unsigned int)nrrds.size(), weight, bin,
                       string_to_nrrd_type(type_.get()), 
		       clamp))
    {
      char *err = biffGetDone(NRRD);
      error(string("Error performing Unu Jhisto: ") +  err);
      free(err);
      return;
    }

    last_nrrdH_ = scinew NrrdData(nout);

    if (nrrds.size())
    {
      if (airIsNaN(range[0]->min) || airIsNaN(range[0]->max))
      {
	NrrdRange *minmax = nrrdRangeNewSet(nrrds_array[0],
                                            nrrdBlind8BitRangeFalse);
	if (airIsNaN(range[0]->min)) range[0]->min = minmax->min;
	if (airIsNaN(range[0]->max)) range[0]->max = minmax->max;
	nrrdRangeNix(minmax);
      }
      nrrdKeyValueAdd(last_nrrdH_->nrrd_, "jhisto_nrrd0_min", 
		      to_string(range[0]->min).c_str());
      nrrdKeyValueAdd(last_nrrdH_->nrrd_, "jhisto_nrrd0_max", 
		      to_string(range[0]->max).c_str());
    }
    
    for (int d=0; d < (int)nrrds.size(); ++d)
    {
      nrrdRangeNix(range[d]);
    }
    delete [] range;
    delete [] nrrds_array;

    delete bin;
    delete min;
    delete max;
    
  }

  send_output_handle("OutputNrrd", last_nrrdH_, true);
}


} // End namespace Teem


