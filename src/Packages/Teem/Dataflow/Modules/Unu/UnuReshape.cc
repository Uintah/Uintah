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
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class UnuReshape : public Module {
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
  : Module("UnuReshape", ctx, Source, "UnuNtoZ", "Teem"),
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
  out->copy_properties(nrrd_handle.get_rep());

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


