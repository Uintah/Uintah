/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  RGBPixelToVector.cc:
 *
 *  Written by:
 *   darbyb
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Network/Ports/ITKDatatypePort.h>

#include <itkVectorCastImageFilter.h>
#include <itkVector.h>
#include <itkRGBPixel.h>


namespace Insight {

using namespace SCIRun;

class RGBPixelToVector : public Module {
public:
  // Declare Ports
  ITKDatatypeHandle inhandle_InputImage_;

  ITKDatatypeHandle outhandle_OutputImage_;

  RGBPixelToVector(GuiContext*);

  virtual ~RGBPixelToVector();

  virtual void execute();

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


RGBPixelToVector::~RGBPixelToVector()
{
}

template<class InputImageType, class OutputImageType>
bool
RGBPixelToVector::run( itk::Object *obj_InputImage) 
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
  send_output_handle("OutputImage", outhandle_OutputImage_, true);
  
  return true;
}


void
RGBPixelToVector::execute()
{
  // check input ports
  if (!get_input_handle("InputImage", inhandle_InputImage_)) return;

  // get input
  itk::Object* data_InputImage = inhandle_InputImage_.get_rep()->data_.GetPointer();
  
  // can we operate on it?
  if(0) { }
  else if(run< itk::Image<itk::RGBPixel<float>, 2>, itk::Image<itk::Vector<float>, 2> >( data_InputImage )) {}
  else if(run< itk::Image<itk::RGBPixel<float>, 3>, itk::Image<itk::Vector<float>, 3> >( data_InputImage )) {}
  else {
    // error
    error("Incorrect input type");
    return;
  } 
}

} // End namespace Insight


