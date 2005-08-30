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
 *  ChooseNrrd.cc: Choose one input Nrrd to be passed downstream
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Handle.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCITeem {
using namespace SCIRun;

class ChooseNrrd : public Module {
private:
  GuiInt port_index_;
  GuiInt usefirstvalid_;
public:
  ChooseNrrd(GuiContext* ctx);
  virtual ~ChooseNrrd();
  virtual void execute();
};

DECLARE_MAKER(ChooseNrrd)
ChooseNrrd::ChooseNrrd(GuiContext* ctx)
  : Module("ChooseNrrd", ctx, Filter, "NrrdData", "Teem"),
    port_index_(ctx->subVar("port-index")),
    usefirstvalid_(ctx->subVar("usefirstvalid"))
{
}

ChooseNrrd::~ChooseNrrd()
{
}

void
ChooseNrrd::execute()
{
  NrrdOPort *onrrd = (NrrdOPort *)get_oport("Nrrd");

  port_range_type range = get_iports("Nrrd");
  if (range.first == range.second)
    return;

  port_map_type::iterator pi = range.first;

  int usefirstvalid = usefirstvalid_.get();

  NrrdIPort *inrrd = 0;
  NrrdDataHandle nrrd;
  
  if (usefirstvalid) {
    // iterate over the connections and use the
    // first valid nrrd
    int idx = 0;
    bool found_valid = false;
    while (pi != range.second) {
      inrrd = (NrrdIPort *)get_iport(idx);
      if (inrrd->get(nrrd) && nrrd != 0) {
	found_valid = true;
	break;
      }
      ++idx;
      ++pi;
    }
    if (!found_valid) {
      error("Didn't find any valid nrrds\n");
      return;
    }
  } else {
    // use the index specified
    int idx=port_index_.get();
    if (idx<0) { error("Can't choose a negative port"); return; }
    while (pi != range.second && idx != 0) { ++pi ; idx--; }
    int port_number=pi->second;
    if (pi == range.second || ++pi == range.second) { 
      error("Selected port index out of range"); return; 
    }

    inrrd = (NrrdIPort *)get_iport(port_number);
    inrrd->get(nrrd);
  }
  
  onrrd->send(nrrd);
}

} // End namespace SCITeem
