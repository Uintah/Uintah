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
 *  IntToUChar.cc:
 *
 *  Written by:
 *   darbyb
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>

#include <itkCastImageFilter.h>
#include <itkVectorCastImageFilter.h>

namespace Insight {

using namespace SCIRun;

class IntToUChar : public Module {
public:
  // Declare Ports
  ITKDatatypeIPort* inport_InputImage_;
  ITKDatatypeHandle inhandle_InputImage_;

  ITKDatatypeOPort* outport_OutputImage_;
  ITKDatatypeHandle outhandle_OutputImage_;

  IntToUChar(GuiContext*);

  virtual ~IntToUChar();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  // Run function will dynamically cast data to determine which
  // instantiation we are working with. The last template type
  // refers to the last template type of the filter intstantiation.
  template<class InputImageType, class OutputImageType > 
  bool run( itk::Object* );
  template<class InputImageType, class OutputImageType > 
  bool run2( itk::Object* );
};


DECLARE_MAKER(IntToUChar)
IntToUChar::IntToUChar(GuiContext* ctx)
  : Module("IntToUChar", ctx, Source, "Converters", "Insight")
{
}

IntToUChar::~IntToUChar(){
}

template<class InputImageType, class OutputImageType>
bool IntToUChar::run( itk::Object *obj_InputImage) 
{
  InputImageType *data_InputImage = dynamic_cast<  InputImageType * >(obj_InputImage);
  
  if( !data_InputImage ) {
    return false;
  }


  typedef typename itk::CastImageFilter< InputImageType, OutputImageType > CasterType;

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


template<class InputImageType, class OutputImageType>
bool IntToUChar::run2( itk::Object *obj_InputImage) 
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

void IntToUChar::execute(){
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
  else if(run< itk::Image<int, 2>, itk::Image<unsigned char, 2> >( data_InputImage )) {} 
  else if(run< itk::Image<int, 3>, itk::Image<unsigned char, 3> >( data_InputImage )) {} 
  else if(run2< itk::Image<itk::Vector<int>, 2>, itk::Image<itk::Vector<unsigned char>, 2> >( data_InputImage )) {} 
  else if(run2< itk::Image<itk::Vector<int>, 3>, itk::Image<itk::Vector<unsigned char>, 3> >( data_InputImage )) {} 
  else {
    // error
    error("Incorrect input type");
    return;
  }
}

void IntToUChar::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight


