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
 *  UnuReshape.cc Superficially change dimension and/or axes sizes.
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

class PSECORESHARE UnuReshape : public Module {
public:
  UnuReshape(GuiContext*);

  virtual ~UnuReshape();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiString       sz_;
};


DECLARE_MAKER(UnuReshape)
UnuReshape::UnuReshape(GuiContext* ctx)
  : Module("UnuReshape", ctx, Source, "Unu", "Teem"),
    inrrd_(0), onrrd_(0),
    sz_(ctx->subVar("sz"))
{
}

UnuReshape::~UnuReshape(){
}

void
 UnuReshape::execute(){
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
  string sizes = sz_.get();
  int szLen = 0;
  char ch;
  int i=0, start=0;
  bool inword = false;
  while (i < (int)sizes.length()) {
    ch = sizes[i];
    if(isspace(ch)) {
      if (inword) {
	szLen++;
	inword = false;
      }
    } else if (i == (int)sizes.length()-1) {
      szLen++;
      inword = false;
    } else {
      if(!inword) 
	inword = true;
    }
    i++;
  }

  if (szLen != nin->dim) {
    error("min coords " + to_string(szLen) + " != nrrd dim " + to_string(nin->dim));
    return;
  }

  int *sz = new int[nin->dim];
  // Size/samples
  i=0, start=0;
  int which = 0, end=0, counter=0;
  inword = false;
  while (i < (int)sizes.length()) {
    ch = sizes[i];
    if(isspace(ch)) {
      if (inword) {
	end = i;
	sz[counter] = (atoi(sizes.substr(start,end-start).c_str()));
	which++;
	counter++;
	inword = false;
      }
    } else if (i == (int)sizes.length()-1) {
      if (!inword) {
	start = i;
      }
      end = i+1;
      sz[counter] = (atoi(sizes.substr(start,end-start).c_str()));
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

  if (nrrdReshape_nva(nout, nin, szLen, sz)) {
    char *err = biffGetDone(NRRD);
    error(string("Error Reshaping nrrd: ") + err);
    free(err);
  }

  delete sz;


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
 UnuReshape::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Teem


