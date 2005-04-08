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
 *  UnuInset.cc Replace a sub-region with a different nrrd. This is functionally
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

class UnuInset : public Module {
public:
  UnuInset(GuiContext*);

  virtual ~UnuInset();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  NrrdIPort*      inrrd_;
  NrrdIPort*      isub_;
  NrrdOPort*      onrrd_;

  GuiString       mins_;
};


DECLARE_MAKER(UnuInset)
UnuInset::UnuInset(GuiContext* ctx)
  : Module("UnuInset", ctx, Source, "UnuAtoM", "Teem"),
    inrrd_(0), isub_(0), onrrd_(0),
    mins_(ctx->subVar("mins"))
{
}

UnuInset::~UnuInset(){
}

void
 UnuInset::execute(){
  NrrdDataHandle nrrd_handle;
  NrrdDataHandle sub_handle;

  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("InputNrrd");
  isub_ = (NrrdIPort *)get_iport("SubRegionNrrd");
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inrrd_->get(nrrd_handle))
    return;
  if (!isub_->get(sub_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty InputNrrd.");
    return;
  }
  if (!sub_handle.get_rep()) {
    error("Empty SubRegionNrrd.");
    return;
  }

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *sub = sub_handle->nrrd;
  Nrrd *nout = nrrdNew();

  // Determine the number of mins given
  string mins = mins_.get();
  int minsLen = 0;
  char ch;
  int i=0, start=0;
  bool inword = false;
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

  if (minsLen != nin->dim) {
    error("min coords " + to_string(minsLen) + " != nrrd dim " + to_string(nin->dim));
    return;
  }

  int *min = new int[nin->dim];
  // Size/samples
  i=0, start=0;
  int which = 0, end=0, counter=0;
  inword = false;
  while (i < (int)mins.length()) {
    ch = mins[i];
    if(isspace(ch)) {
      if (inword) {
	end = i;
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

  if (nrrdInset(nout, nin, sub, min)) {
    char *err = biffGetDone(NRRD);
    error(string("Error Insetting nrrd: ") + err);
    free(err);
  }

  delete min;


  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  // Copy the properties.
  out->copy_properties(nrrd_handle.get_rep());

  // Copy the axis kinds
  for (int i=0; i<nin->dim; i++) {
    nout->axis[i].kind = nin->axis[i].kind;
  }

  onrrd_->send(out);
}

void
 UnuInset::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Teem


