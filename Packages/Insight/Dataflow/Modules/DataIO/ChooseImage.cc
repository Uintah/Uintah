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
 *  ChooseImage.cc:
 *
 *  Written by:
 *   darbyb
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/share/share.h>
#include <Core/Containers/StringUtil.h>

#include <Insight/Dataflow/Ports/ITKDatatypePort.h>

namespace Insight {

using namespace SCIRun;

class PSECORESHARE ChooseImage : public Module {
private:
  GuiInt port_index_;
public:
  ChooseImage(GuiContext*);

  virtual ~ChooseImage();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(ChooseImage)
ChooseImage::ChooseImage(GuiContext* ctx)
  : Module("ChooseImage", ctx, Source, "DataIO", "Insight"),
    port_index_(ctx->subVar("port-index"))
{
}

ChooseImage::~ChooseImage(){
}

void ChooseImage::execute(){
  ITKDatatypeOPort *oimg = (ITKDatatypeOPort *)get_oport("OutputImage");
  if (!oimg) {
    error("Unable to initialize oport 'OututImage'.");
    return;
  }

  update_state(NeedData);

  port_range_type range = get_iports("InputImage");
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
  
  ITKDatatypeIPort *iimage = (ITKDatatypeIPort *)get_iport(port_number);
  if (!iimage) {
    error("Unable to initialize iport '" + to_string(port_number) + "'.");
    return;
  }
  ITKDatatypeHandle image;
  iimage->get(image);
  oimg->send(image);
}

void ChooseImage::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight


