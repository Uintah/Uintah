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
 * MeanImageFilter.cc
 *
 *   Auto Generated File For itk::MeanImageFilter
 *
 */


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>

#include <itkMeanImageFilter.h>

#include <itkCommand.h>

namespace Insight 
{

using namespace SCIRun;

class MeanImageFilter : public Module 
{
public:

  typedef itk::MemberCommand< MeanImageFilter > RedrawCommandType;

  // Filter Declaration
  itk::Object::Pointer filter_;

  // Declare GuiVars
  vector< GuiInt* >  gui_radius_;
  
  GuiInt gui_dimension_;
  bool execute_;
  

  // Declare Ports
  ITKDatatypeIPort* inport_InputImage_;
  ITKDatatypeHandle inhandle_InputImage_;
  int last_InputImage_;

  ITKDatatypeOPort* outport_OutputImage_;
  ITKDatatypeHandle outhandle_OutputImage_;

  
  MeanImageFilter(GuiContext*);

  virtual ~MeanImageFilter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  // Run function will dynamically cast data to determine which
  // instantiation we are working with. The last template type
  // refers to the last template type of the filter intstantiation.
  template<class InputImageType, class OutputImageType > 
  bool run( itk::Object*   );

  // progress bar

  void ProcessEvent(itk::Object * caller, const itk::EventObject & event );
  void ConstProcessEvent(const itk::Object * caller, const itk::EventObject & event );
  void Observe( itk::Object *caller );
  RedrawCommandType::Pointer m_RedrawCommand;

};


template<class InputImageType, class OutputImageType>
bool 
MeanImageFilter::run( itk::Object *obj_InputImage) 
{
  InputImageType *data_InputImage = dynamic_cast<  InputImageType * >(obj_InputImage);
  
  if( !data_InputImage ) {
    return false;
  }

  execute_ = true;
  
  typedef typename itk::MeanImageFilter< InputImageType, OutputImageType > FilterType;

  // Check if filter_ has been created
  // or the input data has changed. If
  // this is the case, set the inputs.

  if(!filter_  || 
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
   
  
  // instantiate any defined objects
  
  typename FilterType::InputSizeType radius;
  
  // clear defined object guis if things aren't in sync
  
  
  if(radius.GetSizeDimension() != gui_dimension_.get()) { 
    gui_dimension_.set(radius.GetSizeDimension());    
  
    // for each defined object, clear gui
  
    gui->execute(id.c_str() + string(" clear_radius_gui"));
    gui->execute(id.c_str() + string(" init_radius_dimensions"));
    
    execute_ = false;
  }
  
  // register GuiVars
  // avoid pushing onto vector each time if not needed
  int start_radius = 0;
  if(gui_radius_.size() > 0) {
    start_radius = gui_radius_.size();
  }

  for(unsigned int i=start_radius; i<radius.GetSizeDimension(); i++) {
    ostringstream str;
    str << "radius" << i;
    
    gui_radius_.push_back(new GuiInt(ctx->subVar(str.str())));

  }

  // set radius values
  for(int i=0; i<radius.GetSizeDimension(); i++) {
    radius[i] = gui_radius_[i]->get();
  }

  dynamic_cast<FilterType* >(filter_.GetPointer())->SetRadius( radius );

  

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


DECLARE_MAKER(MeanImageFilter)

MeanImageFilter::MeanImageFilter(GuiContext* ctx)
  : Module("MeanImageFilter", ctx, Source, "Filters", "Insight"),
     gui_dimension_(ctx->subVar("dimension")), 
     last_InputImage_(-1)
{
  filter_ = 0;

  gui_dimension_.set(0);

  m_RedrawCommand = RedrawCommandType::New();
  m_RedrawCommand->SetCallbackFunction( this, &MeanImageFilter::ProcessEvent );
  m_RedrawCommand->SetCallbackFunction( this, &MeanImageFilter::ConstProcessEvent );

  update_progress(0.0);

}

MeanImageFilter::~MeanImageFilter() 
{
}

void 
MeanImageFilter::execute() 
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
  else if(run< itk::Image<float, 2>, itk::Image<float, 2> >( data_InputImage )) {} 
  else if(run< itk::Image<float, 3>, itk::Image<float, 3> >( data_InputImage )) {} 
  else {
    // error
    error("Incorrect input type");
    return;
  }

}


// Manage a Progress event 
void 
MeanImageFilter::ProcessEvent( itk::Object * caller, const itk::EventObject & event )
{
  if( typeid( itk::ProgressEvent )   ==  typeid( event ) )
  {
    ::itk::ProcessObject::Pointer  process = 
        dynamic_cast< itk::ProcessObject *>( caller );

    const double value = static_cast<double>(process->GetProgress() );
    update_progress( value );
    }

}


// Manage a Progress event 
void 
MeanImageFilter::ConstProcessEvent(const itk::Object * caller, const itk::EventObject & event )
{
  if( typeid( itk::ProgressEvent )   ==  typeid( event ) )
  {
    ::itk::ProcessObject::ConstPointer  process = 
        dynamic_cast< const itk::ProcessObject *>( caller );

    const double value = static_cast<double>(process->GetProgress() );
    update_progress( value );
    }

}


// Manage a Progress event 
void 
MeanImageFilter::Observe( itk::Object *caller )
{
  caller->AddObserver(  itk::ProgressEvent(), m_RedrawCommand.GetPointer() );
  caller->AddObserver(  itk::IterationEvent(), m_RedrawCommand.GetPointer() );
}

void 
MeanImageFilter::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);

}


} // End of namespace Insight
