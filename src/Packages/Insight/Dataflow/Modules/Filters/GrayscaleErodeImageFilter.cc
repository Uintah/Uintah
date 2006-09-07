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
 * GrayscaleErodeImageFilter.cc
 *
 *
 *
 */


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>

#include <itkGrayscaleErodeImageFilter.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkCommand.h>

namespace Insight
{

using namespace SCIRun;

class GrayscaleErodeImageFilter : public Module
{
public:

  typedef itk::MemberCommand< GrayscaleErodeImageFilter > RedrawCommandType;

  // Filter Declaration
  itk::Object::Pointer filter_;


 // Declare GuiVars
  GuiInt gui_radius_;

  bool execute_;
  

  // Declare Ports
  ITKDatatypeIPort* inport_InputImage_;
  ITKDatatypeHandle inhandle_InputImage_;
  int last_InputImage_;

  ITKDatatypeOPort* outport_OutputImage_;
  ITKDatatypeHandle outhandle_OutputImage_;

  
  GrayscaleErodeImageFilter(GuiContext*);

  virtual ~GrayscaleErodeImageFilter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  // Run function will dynamically cast data to determine which
  // instantiation we are working with. The last template type
  // refers to the last template type of the filter intstantiation.
  template<class InputImageType, class OutputImageType >
  bool run( itk::Object* );

  // progress bar

  void ProcessEvent(itk::Object * caller, const itk::EventObject & event );
  void ConstProcessEvent(const itk::Object * caller, const itk::EventObject & event );
  void Observe( itk::Object *caller );
  RedrawCommandType::Pointer m_RedrawCommand;

};


template<class InputImageType, class OutputImageType>
bool
GrayscaleErodeImageFilter::run( itk::Object *obj_InputImage)
{
  InputImageType *data_InputImage = dynamic_cast< InputImageType * >(obj_InputImage);
  
  if( !data_InputImage ) {
    return false;
  }

  execute_ = true;

  typedef itk::BinaryBallStructuringElement< typename InputImageType::PixelType, ::itk::GetImageDimension<InputImageType>::ImageDimension > StructuringElementType;

  typedef typename itk::GrayscaleErodeImageFilter< InputImageType, OutputImageType, StructuringElementType > FilterType;

  // Check if filter_ has been created
  // or the input data has changed. If
  // this is the case, set the inputs.

  if(!filter_ ||
     inhandle_InputImage_->generation != last_InputImage_) {
     
     last_InputImage_ = inhandle_InputImage_->generation;

     // create a new one
     filter_ = FilterType::New();

     // attach observer for progress bar
     Observe( filter_.GetPointer() );

     // set inputs
     
     dynamic_cast<FilterType* >(filter_.GetPointer())->SetInput( data_InputImage );
       
  }

  // reset progress bar
  update_progress(0.0);

  // set filter parameters

  StructuringElementType structuringElement;
  structuringElement.SetRadius( gui_radius_.get() );
  structuringElement.CreateStructuringElement();
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetKernel( structuringElement );
  
  // execute the filter
  
  if (execute_) {
  
  try {

    dynamic_cast<FilterType* >(filter_.GetPointer())->Update();

  } catch ( itk::ExceptionObject & err ) {
     error("ExceptionObject caught!");
     error(err.GetDescription());
  }

  // get filter output
  
  
  ITKDatatype* out_OutputImage_ = scinew ITKDatatype;
  
  out_OutputImage_->data_ = dynamic_cast<FilterType* >(filter_.GetPointer())->GetOutput();

  outhandle_OutputImage_ = out_OutputImage_;

  outport_OutputImage_->send(outhandle_OutputImage_);
  
  }
  

  return true;
}


DECLARE_MAKER(GrayscaleErodeImageFilter)

GrayscaleErodeImageFilter::GrayscaleErodeImageFilter(GuiContext* ctx)
  : Module("GrayscaleErodeImageFilter", ctx, Source, "Filters", "Insight"),
    gui_radius_(get_ctx()->subVar("radius")),
     last_InputImage_(-1)
{
  filter_ = 0;

  m_RedrawCommand = RedrawCommandType::New();
  m_RedrawCommand->SetCallbackFunction( this, &GrayscaleErodeImageFilter::ProcessEvent );
  m_RedrawCommand->SetCallbackFunction( this, &GrayscaleErodeImageFilter::ConstProcessEvent );
}

GrayscaleErodeImageFilter::~GrayscaleErodeImageFilter()
{
}

void
GrayscaleErodeImageFilter::execute()
{
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
  else if(run< itk::Image<float, 2>, itk::Image<float, 2> >( data_InputImage )) { }
  else if(run< itk::Image<float, 3>, itk::Image<float, 3> >( data_InputImage )) { }
  else {
    // error
    error("Incorrect input type");
    return;
  }

}


// Manage a Progress event
void
GrayscaleErodeImageFilter::ProcessEvent( itk::Object * caller, const itk::EventObject & event )
{
  if( typeid( itk::ProgressEvent ) == typeid( event ) )
  {
    ::itk::ProcessObject::Pointer process =
        dynamic_cast< itk::ProcessObject *>( caller );

    const double value = static_cast<double>(process->GetProgress() );
    update_progress( value );
    }

}


// Manage a Progress event
void
GrayscaleErodeImageFilter::ConstProcessEvent(const itk::Object * caller, const itk::EventObject & event )
{
  if( typeid( itk::ProgressEvent ) == typeid( event ) )
  {
    ::itk::ProcessObject::ConstPointer process =
        dynamic_cast< const itk::ProcessObject *>( caller );

    const double value = static_cast<double>(process->GetProgress() );
    update_progress( value );
    }

}


// Manage a Progress event
void
GrayscaleErodeImageFilter::Observe( itk::Object *caller )
{
  caller->AddObserver( itk::ProgressEvent(), m_RedrawCommand.GetPointer() );
  caller->AddObserver( itk::IterationEvent(), m_RedrawCommand.GetPointer() );
}

void
GrayscaleErodeImageFilter::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);

}


} // End of namespace Insight

