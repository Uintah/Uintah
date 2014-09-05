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
 *  UnuShuffle.cc Permute slices along one axis.
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

class PSECORESHARE UnuShuffle : public Module {
public:
  UnuShuffle(GuiContext*);

  virtual ~UnuShuffle();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiString       ordering_;
  GuiInt          axis_;
  GuiInt          inverse_;
};


DECLARE_MAKER(UnuShuffle)
UnuShuffle::UnuShuffle(GuiContext* ctx)
  : Module("UnuShuffle", ctx, Source, "Unu", "Teem"),
    inrrd_(0), onrrd_(0), ordering_(ctx->subVar("ordering")),
    axis_(ctx->subVar("axis")), inverse_(ctx->subVar("inverse"))
{
}

UnuShuffle::~UnuShuffle(){
}

void
 UnuShuffle::execute(){
  NrrdDataHandle nrrd_handle;

  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("InputNrrd");
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inrrd_) {
    error("Unable to initialize iport 'InputNrrd'.");
    return;
  }
  if (!onrrd_) {
    error("Unable to initialize oport 'OutputNrrd'.");
    return;
  }

  if (!inrrd_->get(nrrd_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty InputNrrd.");
    return;
  }

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *nout = nrrdNew();

  // Determine the number of mins given
  string order = ordering_.get();
  int ordLen = 0;
  char ch;
  int i=0, start=0;
  bool inword = false;
  while (i < (int)order.length()) {
    ch = order[i];
    if(isspace(ch)) {
      if (inword) {
	ordLen++;
	inword = false;
      }
    } else if (i == (int)order.length()-1) {
      ordLen++;
      inword = false;
    } else {
      if(!inword) 
	inword = true;
    }
    i++;
  }

  int *ord = new int[ordLen];

  
  i=0, start=0;
  int which = 0, end=0, counter=0;
  inword = false;
  while (i < (int)order.length()) {
    ch = order[i];
    if(isspace(ch)) {
      if (inword) {
	end = i;
	ord[counter] = (atoi(order.substr(start,end-start).c_str()));
	which++;
	counter++;
	inword = false;
      }
    } else if (i == (int)order.length()-1) {
      if (!inword) {
	start = i;
      }
      end = i+1;
      ord[counter] = (atoi(order.substr(start,end-start).c_str()));
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
  
  // error checking
  if (axis_.get() >= nin->dim) {
    error("Axis " + to_string(axis_.get()) + " not in valid range [0," + to_string(nin->dim-1) + "]");
    return;
  }
  if (ordLen != nin->axis[axis_.get()].size) {
    error("Permutation length " + to_string(ordLen) + " != axis " + to_string(axis_.get()) + "'s size of " + to_string(nin->axis[axis_.get()].size));
    return;
  }

  int *iperm;
  int **whichperm;
  if (inverse_.get()) {
    iperm = new int[ordLen];
    if (nrrdInvertPerm(iperm, ord, ordLen)) {
      error("Couldn't compute inverse of given permutation");
      return;
    }
    whichperm = &iperm;
  }
  else {
    whichperm = &ord;
  }

  if (nrrdShuffle(nout, nin, axis_.get(), *whichperm)) {
    char *err = biffGetDone(NRRD);
    error(string("Error Shuffling nrrd: ") + err);
    free(err);
  }

  if (inverse_.get())
    delete iperm;

  delete ord;

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);
  // Copy the properties.
  *((PropertyManager *) out.get_rep()) =
    *((PropertyManager *) nrrd_handle.get_rep());
  
  // Copy the axis kinds
  for (int i=0; i<nin->dim; i++) {
    nout->axis[i].kind = nin->axis[i].kind;
  }

  onrrd_->send(out);

}

void
 UnuShuffle::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Teem


