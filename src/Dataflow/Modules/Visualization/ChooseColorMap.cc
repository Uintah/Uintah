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
 *  ChooseColorMap.cc: Choose one input field to be passed downstream
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
#include <Dataflow/Ports/ColorMapPort.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Handle.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCIRun {

class PSECORESHARE ChooseColorMap : public Module {
private:
  GuiInt port_index_;
public:
  ChooseColorMap(GuiContext* ctx);
  virtual ~ChooseColorMap();
  virtual void execute();
};

DECLARE_MAKER(ChooseColorMap)
ChooseColorMap::ChooseColorMap(GuiContext* ctx)
  : Module("ChooseColorMap", ctx, Filter, "Visualization", "SCIRun"),
    port_index_(ctx->subVar("port-index"))
{
}

ChooseColorMap::~ChooseColorMap()
{
}

void
ChooseColorMap::execute()
{
  ColorMapOPort *ofld = (ColorMapOPort *)get_oport("ColorMap");
  if (!ofld) {
    error("Unable to initialize oport 'ColorMap'.");
    return;
  }

  port_range_type range = get_iports("ColorMap");
  if (range.first == range.second)
    return;

  port_map_type::iterator pi = range.first;
  int idx=port_index_.get();
  if (idx<0) { error("Can't choose a negative port"); return; }
  while (pi != range.second && idx != 0) { ++pi ; idx--; }
  int port_number=pi->second;
  if (pi == range.second || ++pi == range.second) { 
    error("An input pipe is not plugged into the specified port."); return; 
  }
  
  ColorMapIPort *ifield = (ColorMapIPort *)get_iport(port_number);
  if (!ifield) {
    error("Unable to initialize iport '" + to_string(port_number) + "'.");
    return;
  }
  ColorMapHandle field;
  ifield->get(field);
  ofld->send(field);
}

} // End namespace SCIRun
