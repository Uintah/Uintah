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
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Handle.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCITeem {
using namespace SCIRun;

class ChooseNrrd : public Module {
private:
  GuiInt port_index_;
public:
  ChooseNrrd(GuiContext* ctx);
  virtual ~ChooseNrrd();
  virtual void execute();
};

DECLARE_MAKER(ChooseNrrd)
ChooseNrrd::ChooseNrrd(GuiContext* ctx)
  : Module("ChooseNrrd", ctx, Filter, "NrrdData", "Teem"),
    port_index_(ctx->subVar("port-index"))
{
}

ChooseNrrd::~ChooseNrrd()
{
}

void
ChooseNrrd::execute()
{
  NrrdOPort *onrrd = (NrrdOPort *)get_oport("Nrrd");
  if (!onrrd) {
    error("Unable to initialize oport 'Nrrd'.");
    return;
  }

  port_range_type range = get_iports("Nrrd");
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
  
  NrrdIPort *inrrd = (NrrdIPort *)get_iport(port_number);
  if (!inrrd) {
    error("Unable to initialize iport '" + to_string(port_number) + "'.");
    return;
  }
  NrrdDataHandle nrrd;
  inrrd->get(nrrd);
  onrrd->send(nrrd);
}

} // End namespace SCITeem
