/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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

class PSECORESHARE ChooseField : public Module {
private:
  GuiInt port_index_;
public:
  ChooseField(GuiContext* ctx);
  virtual ~ChooseField();
  virtual void execute();
};

DECLARE_MAKER(ChooseField)
ChooseField::ChooseField(GuiContext* ctx)
  : Module("ChooseField", ctx, Filter, "FieldsOther", "SCIRun"),
    port_index_(ctx->subVar("port-index"))
{
}

ChooseField::~ChooseField()
{
}

void
ChooseField::execute()
{
  FieldOPort *ofld = (FieldOPort *)get_oport("Field");
  if (!ofld) {
    error("Unable to initialize oport 'Field'.");
    return;
  }

  update_state(NeedData);

  port_range_type range = get_iports("Field");
  if (range.first == range.second)
    return;

  port_map_type::iterator pi = range.first;
  int idx=port_index_.get();
  if (idx<0) { error("Can't choose a negative port"); return; }
  while (pi != range.second && idx != 0) { ++pi ; idx--; }
  int port_number=pi->second;
  if (pi == range.second || ++pi == range.second) { 
    error("Selected port index out of range"); return; 
  }
  
  FieldIPort *ifield = (FieldIPort *)get_iport(port_number);
  if (!ifield) {
    error("Unable to initialize iport '" + to_string(port_number) + "'.");
    return;
  }
  FieldHandle field;
  ifield->get(field);
  ofld->send(field);
}

} // End namespace SCIRun

