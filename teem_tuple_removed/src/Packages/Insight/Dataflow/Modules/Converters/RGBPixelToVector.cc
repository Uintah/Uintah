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
 *  RGBPixelToVector.cc:
 *
 *  Written by:
 *   darbyb
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/share/share.h>

#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>

#include <itkVectorCastImageFilter.h>
#include <itkVector.h>
#include <itkRGBPixel.h>


namespace Insight {

using namespace SCIRun;

class PSECORESHARE RGBPixelToVector : public Module {
public:
  // Declare Ports
  ITKDatatypeIPort* inport_InputImage_;
  ITKDatatypeHandle inhandle_InputImage_;

  ITKDatatypeOPort* outport_OutputImage_;
  ITKDatatypeHandle outhandle_OutputImage_;

  RGBPixelToVector(GuiContext*);

  virtual ~RGBPixelToVector();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  // Run function will dynamically cast data to determine which
  // instantiation we are working with. The last template type
  // refers to the last template type of the filter intstantiation.
  template<class InputImageType, class OutputImageType > 
  bool run( itk::Object* );
};


DECLARE_MAKER(RGBPixelToVector)
RGBPixelToVector::RGBPixelToVector(GuiContext* ctx)
  : Module("RGBPixelToVector", ctx, Source, "Converters", "Insight")
{
}

RGBPixelToVector::~RGBPixelToVector(){
}

template<class InputImageType, class OutputImageType>
bool RGBPixelToVector::run( itk::Object *obj_InputImage) 
{
  InputImageType *data_InputImage = dynamic_cast<  InputImageType * >(obj_InputImage);
  
  if( !data_InputImage ) {
    return false;
  }


  typedef typename itk::VectorCastImageFilter< InputImageType, OutputImageType > CasterType;

  // create a new one
  typename CasterType::Pointer caster = CasterType::New();
  
  // set inputs 
  caster->SetInput(dynamic_cast<InputImageType* >(inhandle_InputImage_.get_rep()->data_.GetPointer()));   
  

  // execute the filter
  try {
   caster->Update();
  } catch ( itk::ExceptionObject & err ) {
     error("ExceptionObject caught!");
     error(err.GetDescription());
  }

  // get filter output
  
  
  ITKDatatype* out_OutputImage_ = scinew ITKDatatype; 
  
  out_OutputImage_->data_ = caster->GetOutput();
  
  outhandle_OutputImage_ = out_OutputImage_; 
  outport_OutputImage_->send(outhandle_OutputImage_);
  
  return true;
}

void RGBPixelToVector::execute(){
  // check input ports
  inport_InputImage_ = (ITKDatatypeIPort *)get_iport("InputImage");
  if(!inport_InputImage_) {
    error("Unable to initialize iport");
    return;
  }

  inport_InputImage_->get(inhandle_InputImage_);

  if(!inhandle_InputImage_.get_rep()) {
    return;
  }


  // check output ports
  outport_OutputImage_ = (ITKDatatypeOPort *)get_oport("OutputImage");
  if(!outport_OutputImage_) {
    error("Unable to initialize oport");
    return;
  }

  // get input
  itk::Object* data_InputImage = inhandle_InputImage_.get_rep()->data_.GetPointer();
  
  // can we operate on it?
  if(0) { }
  else if(run< itk::Image<itk::RGBPixel<unsigned char>, 2>, itk::Image<itk::Vector<unsigned char>, 2> >( data_InputImage )) {} 
  else if(run< itk::Image<itk::RGBPixel<unsigned short>, 2>, itk::Image<itk::Vector<unsigned short>, 2> >( data_InputImage )) {} 
  else if(run< itk::Image<itk::RGBPixel<double>, 2>, itk::Image<itk::Vector<double>, 2> >( data_InputImage )) {}
  else if(run< itk::Image<itk::RGBPixel<float>, 2>, itk::Image<itk::Vector<float>, 2> >( data_InputImage )) {}
  else if(run< itk::Image<itk::RGBPixel<int>, 2>, itk::Image<itk::Vector<int>, 2> >( data_InputImage )) {}
  else if(run< itk::Image<itk::RGBPixel<unsigned long>, 2>, itk::Image<itk::Vector<unsigned long>, 2> >( data_InputImage )) {}
  else if(run< itk::Image<itk::RGBPixel<char>, 2>, itk::Image<itk::Vector<char>, 2> >( data_InputImage )) {} 
  else if(run< itk::Image<itk::RGBPixel<short>, 2>, itk::Image<itk::Vector<short>, 2> >( data_InputImage )) {} 
  else if(run< itk::Image<itk::RGBPixel<unsigned char>, 3>, itk::Image<itk::Vector<unsigned char>, 3> >( data_InputImage )) {} 
  else if(run< itk::Image<itk::RGBPixel<unsigned short>, 3>, itk::Image<itk::Vector<unsigned short>, 3> >( data_InputImage )) {} 
  else if(run< itk::Image<itk::RGBPixel<double>, 3>, itk::Image<itk::Vector<double>, 3> >( data_InputImage )) {}
  else if(run< itk::Image<itk::RGBPixel<float>, 3>, itk::Image<itk::Vector<float>, 3> >( data_InputImage )) {}
  else if(run< itk::Image<itk::RGBPixel<int>, 3>, itk::Image<itk::Vector<int>, 3> >( data_InputImage )) {}
  else if(run< itk::Image<itk::RGBPixel<unsigned long>, 3>, itk::Image<itk::Vector<unsigned long>, 3> >( data_InputImage )) {}
  else if(run< itk::Image<itk::RGBPixel<char>, 3>, itk::Image<itk::Vector<char>, 3> >( data_InputImage )) {} 
  else if(run< itk::Image<itk::RGBPixel<short>, 3>, itk::Image<itk::Vector<short>, 3> >( data_InputImage )) {} 
  else {
    // error
    error("Incorrect input type");
    return;
  } 
}

void RGBPixelToVector::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight


