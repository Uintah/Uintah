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
 *  Switch.cc:
 *
 *  Written by:
 *   darbyb
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/Insight/share/share.h>

#include <Insight/Dataflow/Ports/ITKDatatypePort.h>

namespace Insight {

using namespace SCIRun;

class InsightSHARE Switch : public Module {
public:

  ITKDatatypeIPort* inport1_;
  ITKDatatypeHandle inhandle1_;

  ITKDatatypeIPort* inport2_;
  ITKDatatypeHandle inhandle2_;

  ITKDatatypeIPort* inport3_;
  ITKDatatypeHandle inhandle3_;

  ITKDatatypeIPort* inport4_;
  ITKDatatypeHandle inhandle4_;

  ITKDatatypeOPort* outport1_;
  ITKDatatypeHandle outhandle1_;

  GuiInt gui_which_port_;

  int which_;

  ITKDatatype* im;

  Switch(GuiContext*);

  virtual ~Switch();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(Switch)
Switch::Switch(GuiContext* ctx)
  : Module("Switch", ctx, Source, "DataIO", "Insight"),
    gui_which_port_(ctx->subVar("which_port"))
{
}

Switch::~Switch(){
}

void
 Switch::execute(){
  // check ports
  inport1_ = (ITKDatatypeIPort *)get_iport("Image1");
  if(!inport1_) {
    error("Unable to initialize iport 'ITKDatatype'");
    return;
  }
  inport2_ = (ITKDatatypeIPort *)get_iport("Image2");
  if(!inport2_) {
    error("Unable to initialize iport 'ITKDatatype'");
    return;
  }
  inport3_ = (ITKDatatypeIPort *)get_iport("Image3");
  if(!inport3_) {
    error("Unable to initialize iport 'ITKDatatype'");
    return;
  }
  inport4_ = (ITKDatatypeIPort *)get_iport("Image4");
  if(!inport4_) {
    error("Unable to initialize iport 'ITKDatatype'");
    return;
  }

  outport1_ = (ITKDatatypeOPort *)get_oport("Image1");
  if(!outport1_) {
    error("Unable to initialize oport 'ITKDatatype'");
    return;
  }

  // determine which input the user wants
  which_ = gui_which_port_.get();

  switch(which_) {
  case 1:
    inport1_->get(inhandle1_);
    if(!inhandle1_.get_rep()) {
      error("Unable to get data in port 1");
      return;
    }
    outhandle1_ = inhandle1_;
    break;
  case 2:
    inport2_->get(inhandle2_);
    outhandle1_ = inhandle2_.get_rep();
    break;
  case 3:
    inport3_->get(inhandle3_);
    outhandle1_ = inhandle3_.get_rep();
    break;
  case 4:
    inport4_->get(inhandle4_);
    outhandle1_ = inhandle4_.get_rep();
    break;
  default:
    break;
  }

  outport1_->send(outhandle1_);  

}

void
 Switch::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight


