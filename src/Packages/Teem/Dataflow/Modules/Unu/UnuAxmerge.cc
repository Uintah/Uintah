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

//    File   : UnuAxmerge.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>


namespace SCITeem {

using namespace SCIRun;

class UnuAxmerge : public Module {
public:
  UnuAxmerge(SCIRun::GuiContext *ctx);
  virtual ~UnuAxmerge();
  virtual void execute();

private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiString       axes_;
};

DECLARE_MAKER(UnuAxmerge)

UnuAxmerge::UnuAxmerge(SCIRun::GuiContext *ctx) : 
  Module("UnuAxmerge", ctx, Filter, "UnuAtoM", "Teem"), 
  axes_(ctx->subVar("axes"))
{
}

UnuAxmerge::~UnuAxmerge() {
}

void 
UnuAxmerge::execute()
{
  NrrdDataHandle nrrd_handle;
  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("InputNrrd");
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inrrd_->get(nrrd_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty input Nrrd.");
    return;
  }

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *nout[2];
  nout[0] = nrrdNew();
  nout[1] = nrrdNew();

   // Determine the number of axes given
  string axes = axes_.get();
  int axesLen = 0;
  char ch;
  int i=0, start=0;
  bool inword = false;
  while (i < (int)axes.length()) {
    ch = axes[i];
    if(isspace(ch)) {
      if (inword) {
	axesLen++;
	inword = false;
      }
    } else if (i == (int)axes.length()-1) {
      axesLen++;
      inword = false;
    } else {
      if(!inword) 
	inword = true;
    }
    i++;
  }

  int *ax = new int[axesLen];
  // Size/samples
  i=0, start=0;
  int which = 0, end=0, counter=0;
  inword = false;
  while (i < (int)axes.length()) {
    ch = axes[i];
    if(isspace(ch)) {
      if (inword) {
	end = i;
	ax[counter] = (atoi(axes.substr(start,end-start).c_str()));
	which++;
	counter++;
	inword = false;
      }
    } else if (i == (int)axes.length()-1) {
      if (!inword) {
	start = i;
      }
      end = i+1;
      ax[counter] = (atoi(axes.substr(start,end-start).c_str()));
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

  if (axesLen > 1) {
    /* sort merge axes into ascending order */
    qsort(ax, axesLen, sizeof(int), nrrdValCompare[nrrdTypeInt]);
  }

  int ni = 0, jj = 0, ii=0;
  for (ii=0; ii<axesLen; ii++) {
    if (nrrdAxesMerge(nout[ni], !ii ? nin : nout[1-ni], ax[ii])) {
      char *err = biffGetDone(NRRD);
      error(string("Error merging axes: ") + err);
      free(err);
    }
    for (jj=ii+1; jj<axesLen; jj++) {
      ax[jj] -= 1;
    }
    ni = 1-ni;
  }

  delete ax;

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout[1-ni];

  NrrdDataHandle out(nrrd);

  // Copy the properties.
  out->copy_properties(nrrd_handle.get_rep());

  onrrd_->send(out);
}


} // End namespace SCITeem
