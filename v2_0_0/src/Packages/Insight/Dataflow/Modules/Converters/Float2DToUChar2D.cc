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
 *  Float2DToUChar2D.cc:
 *
 *  Written by:
 *   darbyb
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/Insight/share/share.h>

#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>

#include <itkCastImageFilter.h>

namespace Insight {

using namespace SCIRun;

class InsightSHARE Float2DToUChar2D : public Module {
  typedef itk::CastImageFilter< itk::Image<float,2>, itk::Image<unsigned char,2> > CastImageType;
public:
  CastImageType::Pointer to_uchar_;  

  // Declare Ports
  ITKDatatypeIPort* inport1_;
  ITKDatatypeHandle inhandle1_;

  ITKDatatypeOPort* outport1_;
  ITKDatatypeHandle outhandle1_;

  Float2DToUChar2D(GuiContext*);

  virtual ~Float2DToUChar2D();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);  
};


DECLARE_MAKER(Float2DToUChar2D)
Float2DToUChar2D::Float2DToUChar2D(GuiContext* ctx)
  : Module("Float2DToUChar2D", ctx, Source, "Converters", "Insight")
{
  to_uchar_ = CastImageType::New();
}

Float2DToUChar2D::~Float2DToUChar2D(){
}

void Float2DToUChar2D::execute(){
  // check input ports
  inport1_ = (ITKDatatypeIPort *)get_iport("InputImage");
  if(!inport1_) {
    error("Unable to initialize iport");
    return;
  }

  inport1_->get(inhandle1_);
  if(!inhandle1_.get_rep()) {
    return;
  }

  // check output ports
  outport1_ = (ITKDatatypeOPort *)get_oport("OutputImage");
  if(!outport1_) {
    error("Unable to initialize oport");
    return;
  }

  to_uchar_->SetInput(dynamic_cast<itk::Image<float, 2>* >(inhandle1_.get_rep()->data_.GetPointer()));
  
  try {
    to_uchar_->Update();
  } catch ( itk::ExceptionObject & err ) {
     std::cerr << "ExceptionObject caught!" << std::endl;
     std::cerr << err << std::endl;
  }
  
  if(!outhandle1_.get_rep())
  {
    ITKDatatype* im = scinew ITKDatatype;
    im->data_ = to_uchar_->GetOutput();
    outhandle1_ = im; 
  }
  
  // send the data downstream
  outport1_->send(outhandle1_);
}

void Float2DToUChar2D::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight


