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
 * NeighborhoodConnectedImageFilter.cc
 *
 *   Auto Generated File For itk::NeighborhoodConnectedImageFilter
 *
 */


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>

#include <itkNeighborhoodConnectedImageFilter.h>

#include <itkCommand.h>

namespace Insight 
{

using namespace SCIRun;

class NeighborhoodConnectedImageFilter : public Module 
{
public:

  typedef itk::MemberCommand< NeighborhoodConnectedImageFilter > RedrawCommandType;

  // Filter Declaration
  itk::Object::Pointer filter_;

  // Declare GuiVars
  vector< GuiInt* >  gui_radius_;
  vector< GuiInt* >  gui_seed_point_;
  GuiDouble  gui_replace_value_;
  GuiDouble  gui_lower_threshold_;
  GuiDouble  gui_upper_threshold_;
  
  GuiInt gui_dimension_;
  bool execute_;
  

  // Declare Ports
  ITKDatatypeIPort* inport_InputImage_;
  ITKDatatypeHandle inhandle_InputImage_;
  int last_InputImage_;

  ITKDatatypeOPort* outport_OutputImge_;
  ITKDatatypeHandle outhandle_OutputImge_;

  
  NeighborhoodConnectedImageFilter(GuiContext*);

  virtual ~NeighborhoodConnectedImageFilter();

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
NeighborhoodConnectedImageFilter::run( itk::Object *obj_InputImage) 
{
  InputImageType *data_InputImage = dynamic_cast<  InputImageType * >(obj_InputImage);
  
  if( !data_InputImage ) {
    return false;
  }

  execute_ = true;
  
  typedef typename itk::NeighborhoodConnectedImageFilter< InputImageType, OutputImageType > FilterType;

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
  
  typename FilterType::InputImageSizeType radius;
  
  typename FilterType::IndexType seed_point;
  
  // clear defined object guis if things aren't in sync
  
  
  if(radius.GetSizeDimension() != gui_dimension_.get()) { 
    gui_dimension_.set(radius.GetSizeDimension());    
  
    // for each defined object, clear gui
  
    get_gui()->execute(get_id().c_str() + string(" clear_radius_gui"));
    get_gui()->execute(get_id().c_str() + string(" init_radius_dimensions"));
    
    get_gui()->execute(get_id().c_str() + string(" clear_seed_point_gui"));
    get_gui()->execute(get_id().c_str() + string(" init_seed_point_dimensions"));
    
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
    
    gui_radius_.push_back(new GuiInt(get_ctx()->subVar(str.str())));

  }

  // set radius values
  for(int i=0; i<radius.GetSizeDimension(); i++) {
    radius[i] = gui_radius_[i]->get();
  }

  dynamic_cast<FilterType* >(filter_.GetPointer())->SetRadius( radius );

  
  // register GuiVars
  // avoid pushing onto vector each time if not needed
  int start_seed_point = 0;
  if(gui_seed_point_.size() > 0) {
    start_seed_point = gui_seed_point_.size();
  }

  for(unsigned int i=start_seed_point; i<seed_point.GetIndexDimension(); i++) {
    ostringstream str;
    str << "seed_point" << i;
    
    gui_seed_point_.push_back(new GuiInt(get_ctx()->subVar(str.str())));

  }

  // set seed_point values
  for(int i=0; i<seed_point.GetIndexDimension(); i++) {
    seed_point[i] = gui_seed_point_[i]->get();
  }

  dynamic_cast<FilterType* >(filter_.GetPointer())->AddSeed( seed_point );

  
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetReplaceValue( gui_replace_value_.get() ); 
  
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetLower( gui_lower_threshold_.get() ); 
  
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetUpper( gui_upper_threshold_.get() ); 
  

  // execute the filter
  
  if (execute_) {
  
  try {

    dynamic_cast<FilterType* >(filter_.GetPointer())->Update();

  } catch ( itk::ExceptionObject & err ) {
     error("ExceptionObject caught!");
     error(err.GetDescription());
  }

  // get filter output
  
  
  ITKDatatype* out_OutputImge_ = scinew ITKDatatype; 
  
  out_OutputImge_->data_ = dynamic_cast<FilterType* >(filter_.GetPointer())->GetOutput();
  
  outhandle_OutputImge_ = out_OutputImge_; 
  outport_OutputImge_->send(outhandle_OutputImge_);
  
  }
  

  return true;
}


DECLARE_MAKER(NeighborhoodConnectedImageFilter)

NeighborhoodConnectedImageFilter::NeighborhoodConnectedImageFilter(GuiContext* ctx)
  : Module("NeighborhoodConnectedImageFilter", ctx, Source, "Filters", "Insight"),
     gui_replace_value_(get_ctx()->subVar("replace_value")),
     gui_lower_threshold_(get_ctx()->subVar("lower_threshold")),
     gui_upper_threshold_(get_ctx()->subVar("upper_threshold")),
     gui_dimension_(get_ctx()->subVar("dimension")), 
     last_InputImage_(-1)
{
  filter_ = 0;

  gui_dimension_.set(0);

  m_RedrawCommand = RedrawCommandType::New();
  m_RedrawCommand->SetCallbackFunction( this, &NeighborhoodConnectedImageFilter::ProcessEvent );
  m_RedrawCommand->SetCallbackFunction( this, &NeighborhoodConnectedImageFilter::ConstProcessEvent );

  update_progress(0.0);

}

NeighborhoodConnectedImageFilter::~NeighborhoodConnectedImageFilter() 
{
}

void 
NeighborhoodConnectedImageFilter::execute() 
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
  outport_OutputImge_ = (ITKDatatypeOPort *)get_oport("OutputImge");
  if(!outport_OutputImge_) {
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
NeighborhoodConnectedImageFilter::ProcessEvent( itk::Object * caller, const itk::EventObject & event )
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
NeighborhoodConnectedImageFilter::ConstProcessEvent(const itk::Object * caller, const itk::EventObject & event )
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
NeighborhoodConnectedImageFilter::Observe( itk::Object *caller )
{
  caller->AddObserver(  itk::ProgressEvent(), m_RedrawCommand.GetPointer() );
  caller->AddObserver(  itk::IterationEvent(), m_RedrawCommand.GetPointer() );
}

void 
NeighborhoodConnectedImageFilter::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);

}


} // End of namespace Insight
