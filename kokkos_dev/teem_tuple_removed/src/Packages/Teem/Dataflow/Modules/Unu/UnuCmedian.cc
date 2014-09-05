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
//    File   : UnuCmedian.cc
//    Author : Martin Cole
//    Date   : Mon Aug 25 10:13:16 2003

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

class UnuCmedian : public Module {
public:
  UnuCmedian(SCIRun::GuiContext *ctx);
  virtual ~UnuCmedian();
  virtual void execute();

private:


  Nrrd* do_filter(Nrrd *nin);
  bool is_scalar(const string& s) const;

  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiInt          mode_;
  GuiInt          radius_;
  GuiDouble       weight_;
  GuiInt          bins_;
  GuiInt          pad_;
};

DECLARE_MAKER(UnuCmedian)

UnuCmedian::UnuCmedian(SCIRun::GuiContext *ctx) : 
  Module("UnuCmedian", ctx, Filter, "Unu", "Teem"), 
  mode_(ctx->subVar("mode")),
  radius_(ctx->subVar("radius")),
  weight_(ctx->subVar("weight")),
  bins_(ctx->subVar("bins")),
  pad_(ctx->subVar("pad"))
{
}

UnuCmedian::~UnuCmedian() {
}


Nrrd*
UnuCmedian::do_filter(Nrrd *nin)
{
  reset_vars();
  Nrrd *ntmp, *nout;
  nout = nrrdNew();
  ntmp = nin;
//   if (pad_.get()) {
//     ntmp = nrrdNew();
//     if (nrrdSimplePad(ntmp, nin, radius_.get(), nrrdBoundaryBleed)) {
//       char *err = biffGetDone(NRRD);
//       error(string("Error padding: ") + err);
//       free(err);
//       return 0;
//     }
//   }
//   else {
//     ntmp = nin;
//   }

  if (nrrdCheapMedian(nout, ntmp, pad_.get(), mode_.get(), radius_.get(), 
		      weight_.get(), bins_.get())) {
    char *err = biffGetDone(NRRD);
    error(string("Error doing cheap median: ") + err);
    free(err);
    return 0;
  }
  
//   if (pad_.get()) {
//     if (nrrdSimpleCrop(ntmp, nout, radius_.get())) {
//       char *err = biffGetDone(NRRD);
//       error(string("Error cropping: ") + err);
//       free(err);
//       return 0;
//     }
//     nrrdNuke(nout);
//     return ntmp;
//   }
  return nout;
}

bool
UnuCmedian::is_scalar(const string& s) const
{
  const string scalar(":Scalar");
  return (s.find(scalar) < s.size());
}

void 
UnuCmedian::execute()
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

  Nrrd *nin = nrrd_handle->nrrd;
  NrrdDataHandle nsend(0);

  // loop over the tuple axis, and do the median filtering for 
  // each scalar set independently, copy non scalar sets unchanged.
  //vector<string> elems;
  //nrrd_handle->get_tuple_indecies(elems);


  int min[NRRD_DIM_MAX], max[NRRD_DIM_MAX];
  for (int i = 0; i < nrrd_handle->nrrd->dim; i++)
  {
    min[i] = 0;
    max[i] = nrrd_handle->nrrd->axis[i].size - 1;
  }


  //! Slice a scalar out of the tuple axis and filter it. So for Vectors
  //! and Tensors, a component wise filtering occurs.

  if(nrrdKindSize(nrrd_handle->nrrd->axis[0].kind) > 1) {
    vector<Nrrd*> out;
    for (int i = 0; i < nrrd_handle->nrrd->axis[0].size; i++) 
    { 
      Nrrd *sliced = nrrdNew();
      if (nrrdSlice(sliced, nin, 0, i)) {
	char *err = biffGetDone(NRRD);
	error(string("Trouble with slice: ") + err);
	free(err);
      }
      
      Nrrd *nout_filtered;
      nout_filtered = do_filter(sliced);
      if (!nout_filtered) {
	error("Error filtering, returning");
	return;
      }
      out.push_back(nout_filtered);
    }
    // Join the filtered nrrds along the first axis
    NrrdData *nrrd_joined = scinew NrrdData;
    nrrd_joined->nrrd = nrrdNew();
    
    if (nrrdJoin(nrrd_joined->nrrd, &out[0], out.size(), 0, 1)) {
      char *err = biffGetDone(NRRD);
      error(string("Join Error: ") +  err);
      free(err);
      return;
    }
    //nrrd_joined->nrrd->axis[0].label = strdup(nin->axis[0].label);
    //nrrd_joined->copy_sci_data(*nrrd_handle.get_rep());
    onrrd_->send(NrrdDataHandle(nrrd_joined));
  } else {
    Nrrd *nout_filtered;
    nout_filtered = do_filter(nrrd_handle->nrrd);
    if (!nout_filtered) {
      error("Error filtering, returning");
      return;
    }
    NrrdData *nrrd_out = scinew NrrdData;
    nrrd_out->nrrd = nout_filtered;
    //nrrd_out->copy_sci_data(*nrrd_handle.get_rep());
    onrrd_->send(NrrdDataHandle(nrrd_out));
  }
}

} // End namespace SCITeem
