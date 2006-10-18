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
 * IsolatedConnectedImageFilter.cc
 *
 *   Auto Generated File For itk::IsolatedConnectedImageFilter
 *
 */


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>

#include <itkIsolatedConnectedImageFilter.h>

#include <itkCommand.h>

namespace Insight 
{

using namespace SCIRun;

class IsolatedConnectedImageFilter : public Module 
{
public:

  typedef itk::MemberCommand< IsolatedConnectedImageFilter > RedrawCommandType;

  // Filter Declaration
  itk::Object::Pointer filter_;

  // Declare GuiVars
  vector< GuiInt* >  gui_seed_point_1_;
  vector< GuiInt* >  gui_seed_point_2_;
  GuiDouble  gui_replace_value_;
  GuiDouble  gui_lower_threshold_;
  GuiDouble  gui_upper_value_limit_;
  GuiDouble  gui_isolated_value_tolerance_;
  
  GuiInt gui_dimension_;
  bool execute_;

  // Declare Ports
  ITKDatatypeHandle inhandle_InputImage_;
  int last_InputImage_;

  ITKDatatypeHandle outhandle_OutputImage_;
  
  IsolatedConnectedImageFilter(GuiContext*);

  virtual ~IsolatedConnectedImageFilter();

  virtual void execute();

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
IsolatedConnectedImageFilter::run( itk::Object *obj_InputImage) 
{
  InputImageType *data_InputImage = dynamic_cast<  InputImageType * >(obj_InputImage);
  
  if( !data_InputImage ) {
    return false;
  }

  execute_ = true;
  
  typedef typename itk::IsolatedConnectedImageFilter< InputImageType, OutputImageType > FilterType;

  // Check if filter_ has been created
  // or the input data has changed. If
  // this is the case, set the inputs.

  if(!filter_  || 
     inhandle_InputImage_->generation != last_InputImage_)
  {
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
  typename FilterType::IndexType seed_point_1;
  
  typename FilterType::IndexType seed_point_2;
  
  // clear defined object guis if things aren't in sync
  if((int)seed_point_1.GetIndexDimension() != gui_dimension_.get()) { 
    gui_dimension_.set(seed_point_1.GetIndexDimension());    
  
    // for each defined object, clear gui
  
    get_gui()->execute(get_id().c_str() + string(" clear_seed_point_1_gui"));
    get_gui()->execute(get_id().c_str() + string(" init_seed_point_1_dimensions"));
    
    get_gui()->execute(get_id().c_str() + string(" clear_seed_point_2_gui"));
    get_gui()->execute(get_id().c_str() + string(" init_seed_point_2_dimensions"));
    
    execute_ = false;
  }
  
  // register GuiVars
  // avoid pushing onto vector each time if not needed
  int start_seed_point_1 = 0;
  if(gui_seed_point_1_.size() > 0) {
    start_seed_point_1 = gui_seed_point_1_.size();
  }

  for(unsigned int i=start_seed_point_1; i<seed_point_1.GetIndexDimension(); i++) {
    ostringstream str;
    str << "seed_point_1" << i;
    
    gui_seed_point_1_.push_back(new GuiInt(get_ctx()->subVar(str.str())));

  }

  // set seed_point_1 values
  for(unsigned int i=0; i<seed_point_1.GetIndexDimension(); i++) {
    seed_point_1[i] = gui_seed_point_1_[i]->get();
  }

  dynamic_cast<FilterType* >(filter_.GetPointer())->SetSeed1( seed_point_1 );
  
  // register GuiVars
  // avoid pushing onto vector each time if not needed
  int start_seed_point_2 = 0;
  if(gui_seed_point_2_.size() > 0) {
    start_seed_point_2 = gui_seed_point_2_.size();
  }

  for(unsigned int i=start_seed_point_2; i<seed_point_2.GetIndexDimension(); i++) {
    ostringstream str;
    str << "seed_point_2" << i;
    
    gui_seed_point_2_.push_back(new GuiInt(get_ctx()->subVar(str.str())));

  }

  // set seed_point_2 values
  for(unsigned int i=0; i<seed_point_2.GetIndexDimension(); i++) {
    seed_point_2[i] = gui_seed_point_2_[i]->get();
  }

  dynamic_cast<FilterType* >(filter_.GetPointer())->SetSeed2( seed_point_2 );

  
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetReplaceValue( gui_replace_value_.get() ); 
  
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetLower( gui_lower_threshold_.get() ); 
  
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetUpperValueLimit( gui_upper_value_limit_.get() ); 
  
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetIsolatedValueTolerance( gui_isolated_value_tolerance_.get() ); 
  

  // execute the filter
  if (execute_)
  {
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
    send_output_handle("OutputImage", outhandle_OutputImage_, true);
  }

  return true;
}


DECLARE_MAKER(IsolatedConnectedImageFilter)

IsolatedConnectedImageFilter::IsolatedConnectedImageFilter(GuiContext* ctx)
  : Module("IsolatedConnectedImageFilter", ctx, Source, "Filters", "Insight"),
     gui_replace_value_(get_ctx()->subVar("replace_value")),
     gui_lower_threshold_(get_ctx()->subVar("lower_threshold")),
     gui_upper_value_limit_(get_ctx()->subVar("upper_value_limit")),
     gui_isolated_value_tolerance_(get_ctx()->subVar("isolated_value_tolerance")),
     gui_dimension_(get_ctx()->subVar("dimension")), 
     last_InputImage_(-1)
{
  filter_ = 0;

  gui_dimension_.set(0);

  m_RedrawCommand = RedrawCommandType::New();
  m_RedrawCommand->SetCallbackFunction( this, &IsolatedConnectedImageFilter::ProcessEvent );
  m_RedrawCommand->SetCallbackFunction( this, &IsolatedConnectedImageFilter::ConstProcessEvent );
}


IsolatedConnectedImageFilter::~IsolatedConnectedImageFilter() 
{
}


void 
IsolatedConnectedImageFilter::execute() 
{
  // check input ports
  if (!get_input_handle("InputImage", inhandle_InputImage_)) return;

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
IsolatedConnectedImageFilter::ProcessEvent( itk::Object * caller, const itk::EventObject & event )
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
IsolatedConnectedImageFilter::ConstProcessEvent(const itk::Object * caller, const itk::EventObject & event )
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
IsolatedConnectedImageFilter::Observe( itk::Object *caller )
{
  caller->AddObserver(  itk::ProgressEvent(), m_RedrawCommand.GetPointer() );
  caller->AddObserver(  itk::IterationEvent(), m_RedrawCommand.GetPointer() );
}


} // End of namespace Insight
