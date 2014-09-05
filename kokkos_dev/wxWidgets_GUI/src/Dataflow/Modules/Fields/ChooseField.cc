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
 *  ChooseField.cc: Choose one input field to be passed downstream
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
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Handle.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCIRun {

class ChooseField : public Module {
private:
  GuiInt port_index_;
  GuiInt usefirstvalid_;
public:
  ChooseField(GuiContext* ctx);
  virtual ~ChooseField();
  virtual void execute();
};

DECLARE_MAKER(ChooseField)
ChooseField::ChooseField(GuiContext* ctx)
  : Module("ChooseField", ctx, Filter, "FieldsOther", "SCIRun"),
    port_index_(ctx->subVar("port-index")),
    usefirstvalid_(ctx->subVar("usefirstvalid"))
{
}

ChooseField::~ChooseField()
{
}

void
ChooseField::execute()
{
  FieldOPort *ofld = (FieldOPort *)get_oport("Field");

  update_state(NeedData);

  port_range_type range = get_iports("Field");
  if (range.first == range.second)
    return;

  port_map_type::iterator pi = range.first;
  
  int usefirstvalid = usefirstvalid_.get();

  FieldIPort *ifield = 0;
  FieldHandle field;
  
  if (usefirstvalid) {
    // iterate over the connections and use the
    // first valid field
    int idx = 0;
    bool found_valid = false;
    while (pi != range.second) {
      ifield = (FieldIPort *)get_iport(idx);
      if (ifield->get(field) && field != 0) {
	found_valid = true;
	break;
      }
      ++idx;
      ++pi;
    }
    if (!found_valid) {
      error("Didn't find any valid fields.");
      return;
    }
  } else {
    // use the index specified
    int idx=port_index_.get();
    if (idx<0) { error("Can't choose a negative port."); return; }
    while (pi != range.second && idx != 0) { ++pi ; idx--; }
    int port_number=pi->second;
    if (pi == range.second || ++pi == range.second) { 
      error("Selected port index out of range."); return; 
    }

    ifield = (FieldIPort *)get_iport(port_number);
    ifield->get(field);
  }
  
  ofld->send_and_dereference(field);
}

} // End namespace SCIRun

