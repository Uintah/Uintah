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
 * ThresholdSegmentationLevelSetImageFilter.cc
 *
 *   Auto Generated File For itk::ThresholdSegmentationLevelSetImageFilter
 *
 */


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/Insight/share/share.h>
#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>

#include <itkThresholdSegmentationLevelSetImageFilter.h>

#include <itkCommand.h>

namespace Insight 
{

using namespace SCIRun;

class InsightSHARE ThresholdSegmentationLevelSetImageFilter : public Module 
{
public:

  typedef itk::MemberCommand< ThresholdSegmentationLevelSetImageFilter > RedrawCommandType;

  // Filter Declaration
  itk::Object::Pointer filter_;

  // Declare GuiVars
  GuiDouble  gui_lower_threshold_;
  GuiDouble  gui_upper_threshold_;
  GuiDouble  gui_curvature_scaling_;
  GuiDouble  gui_propagation_scaling_;
  GuiDouble  gui_edge_weight_;
  GuiInt  gui_max_iterations_;
  GuiDouble  gui_max_rms_change_;
  GuiInt  gui_reverse_expansion_direction_;
  GuiDouble  gui_isovalue_;
  GuiInt  gui_smoothing_iterations_;
  GuiDouble  gui_smoothing_time_step_;
  GuiDouble  gui_smoothing_conductance_;
  
  GuiInt gui_update_OutputImage_;
  GuiInt gui_update_iters_OutputImage_;
  GuiInt gui_stop_;

  bool execute_;
  

  // Declare Ports
  ITKDatatypeIPort* inport_SeedImage_;
  ITKDatatypeHandle inhandle_SeedImage_;
  int last_SeedImage_;

  ITKDatatypeIPort* inport_FeatureImage_;
  ITKDatatypeHandle inhandle_FeatureImage_;
  int last_FeatureImage_;

  ITKDatatypeOPort* outport_OutputImage_;
  ITKDatatypeHandle outhandle_OutputImage_;

  ITKDatatypeOPort* outport_SpeedImage_;
  ITKDatatypeHandle outhandle_SpeedImage_;

  
  ThresholdSegmentationLevelSetImageFilter(GuiContext*);

  virtual ~ThresholdSegmentationLevelSetImageFilter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  // Run function will dynamically cast data to determine which
  // instantiation we are working with. The last template type
  // refers to the last template type of the filter intstantiation.
  template<class ImageType, class FeatureImageType > 
  bool run( itk::Object*  , itk::Object*   );

  // progress bar

  void update_after_iteration();

  void ProcessEvent(itk::Object * caller, const itk::EventObject & event );
  void ConstProcessEvent(const itk::Object * caller, const itk::EventObject & event );
  void Observe( itk::Object *caller );
  RedrawCommandType::Pointer m_RedrawCommand;

  template<class ImageType, class FeatureImageType>
  bool do_it_OutputImage();
  unsigned int iterationCounter_OutputImage;

  template<class ImageType, class FeatureImageType>
  bool do_it_stop();

};


template<class ImageType, class FeatureImageType>
bool 
ThresholdSegmentationLevelSetImageFilter::run( itk::Object *obj_SeedImage, itk::Object *obj_FeatureImage) 
{
  ImageType *data_SeedImage = dynamic_cast<  ImageType * >(obj_SeedImage);
  
  if( !data_SeedImage ) {
    return false;
  }
  FeatureImageType *data_FeatureImage = dynamic_cast<  FeatureImageType * >(obj_FeatureImage);
  
  if( !data_FeatureImage ) {
    return false;
  }

  typedef typename itk::ThresholdSegmentationLevelSetImageFilter< ImageType, FeatureImageType > FilterType;

  // Check if filter_ has been created
  // or the input data has changed. If
  // this is the case, set the inputs.

  if(filter_ == 0 || 
     inhandle_SeedImage_->generation != last_SeedImage_ || 
     inhandle_FeatureImage_->generation != last_FeatureImage_) {
     
     last_SeedImage_ = inhandle_SeedImage_->generation;
     last_FeatureImage_ = inhandle_FeatureImage_->generation;

     // create a new one
     filter_ = FilterType::New();

     // attach observer for progress bar
     Observe( filter_.GetPointer() );

     // set inputs 
     
     dynamic_cast<FilterType* >(filter_.GetPointer())->SetInput( data_SeedImage );
  
     dynamic_cast<FilterType* >(filter_.GetPointer())->SetFeatureImage( data_FeatureImage );
       
  }

  // reset progress bar
  update_progress(0.0);

  // set filter parameters
   
  dynamic_cast<FilterType* >(filter_.GetPointer())->AbortGenerateDataOff(); 
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetLowerThreshold( gui_lower_threshold_.get() ); 
  
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetUpperThreshold( gui_upper_threshold_.get() ); 
  
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetCurvatureScaling( gui_curvature_scaling_.get() ); 
  
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetPropagationScaling( gui_propagation_scaling_.get() ); 
  
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetEdgeWeight( gui_edge_weight_.get() ); 
  
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetNumberOfIterations( gui_max_iterations_.get() ); 
  
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetMaximumRMSError( gui_max_rms_change_.get() ); 
  
  if( gui_reverse_expansion_direction_.get() ) {
    dynamic_cast<FilterType* >(filter_.GetPointer())->ReverseExpansionDirectionOn( );   
  } 
  else { 
    dynamic_cast<FilterType* >(filter_.GetPointer())->ReverseExpansionDirectionOff( );
  }  
  
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetIsoSurfaceValue( gui_isovalue_.get() ); 
  
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetSmoothingIterations( gui_smoothing_iterations_.get() ); 
  
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetSmoothingTimeStep( gui_smoothing_time_step_.get() ); 
  
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetSmoothingConductance( gui_smoothing_conductance_.get() ); 
  

  // execute the filter
  
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
  
  
  ITKDatatype* out_SpeedImage_ = scinew ITKDatatype; 
  
  out_SpeedImage_->data_ = const_cast<ImageType*  >(dynamic_cast<FilterType* >(filter_.GetPointer())->GetSpeedImage());
  
  outhandle_SpeedImage_ = out_SpeedImage_; 
  outport_SpeedImage_->send(outhandle_SpeedImage_);
  

  return true;
}


DECLARE_MAKER(ThresholdSegmentationLevelSetImageFilter)

ThresholdSegmentationLevelSetImageFilter::ThresholdSegmentationLevelSetImageFilter(GuiContext* ctx)
  : Module("ThresholdSegmentationLevelSetImageFilter", ctx, Source, "Filters", "Insight"),
     gui_lower_threshold_(ctx->subVar("lower_threshold")),
     gui_upper_threshold_(ctx->subVar("upper_threshold")),
     gui_curvature_scaling_(ctx->subVar("curvature_scaling")),
     gui_propagation_scaling_(ctx->subVar("propagation_scaling")),
     gui_edge_weight_(ctx->subVar("edge_weight")),
     gui_max_iterations_(ctx->subVar("max_iterations")),
     gui_max_rms_change_(ctx->subVar("max_rms_change")),
     gui_reverse_expansion_direction_(ctx->subVar("reverse_expansion_direction")),
     gui_isovalue_(ctx->subVar("isovalue")),
     gui_smoothing_iterations_(ctx->subVar("smoothing_iterations")),
     gui_smoothing_time_step_(ctx->subVar("smoothing_time_step")),
     gui_smoothing_conductance_(ctx->subVar("smoothing_conductance")),
     gui_update_OutputImage_(ctx->subVar("update_OutputImage")),
     gui_update_iters_OutputImage_(ctx->subVar("update_iters_OutputImage")), 
     gui_stop_(ctx->subVar("stop")),
     last_SeedImage_(-1), 
     last_FeatureImage_(-1)
{
  filter_ = 0;


  m_RedrawCommand = RedrawCommandType::New();
  m_RedrawCommand->SetCallbackFunction( this, &ThresholdSegmentationLevelSetImageFilter::ProcessEvent );
  m_RedrawCommand->SetCallbackFunction( this, &ThresholdSegmentationLevelSetImageFilter::ConstProcessEvent );

  iterationCounter_OutputImage = 0;

  update_progress(0.0);

}

ThresholdSegmentationLevelSetImageFilter::~ThresholdSegmentationLevelSetImageFilter() 
{
}

void 
ThresholdSegmentationLevelSetImageFilter::execute() 
{
  // check input ports
  inport_SeedImage_ = (ITKDatatypeIPort *)get_iport("SeedImage");
  if(!inport_SeedImage_) {
    error("Unable to initialize iport");
    return;
  }

  inport_SeedImage_->get(inhandle_SeedImage_);

  if(!inhandle_SeedImage_.get_rep()) {
    return;
  }

  inport_FeatureImage_ = (ITKDatatypeIPort *)get_iport("FeatureImage");
  if(!inport_FeatureImage_) {
    error("Unable to initialize iport");
    return;
  }

  inport_FeatureImage_->get(inhandle_FeatureImage_);

  if(!inhandle_FeatureImage_.get_rep()) {
    return;
  }


  // check output ports
  outport_OutputImage_ = (ITKDatatypeOPort *)get_oport("OutputImage");
  if(!outport_OutputImage_) {
    error("Unable to initialize oport");
    return;
  }
  outport_SpeedImage_ = (ITKDatatypeOPort *)get_oport("SpeedImage");
  if(!outport_SpeedImage_) {
    error("Unable to initialize oport");
    return;
  }

  itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);

  iterationCounter_OutputImage = 0;	
  gui_update_OutputImage_.reset();
  gui_update_iters_OutputImage_.reset();

  // get input
  itk::Object* data_SeedImage = inhandle_SeedImage_.get_rep()->data_.GetPointer();
  itk::Object* data_FeatureImage = inhandle_FeatureImage_.get_rep()->data_.GetPointer();
  
  // set stop
  gui_stop_.set(0);

  // can we operate on it?
  if(0) { } 
  else if(run< itk::Image<float, 2>, itk::Image<float, 2> >( data_SeedImage, data_FeatureImage )) { } 
  else if(run< itk::Image<float, 3>, itk::Image<float, 3> >( data_SeedImage, data_FeatureImage )) { }
  else {
    // error
    error("Incorrect input type");
    return;
  }

}


// Manage a Progress event 
void 
ThresholdSegmentationLevelSetImageFilter::ProcessEvent( itk::Object * caller, const itk::EventObject & event )
{
  if (gui_stop_.get() == 1) return;

  if( typeid( itk::ProgressEvent )   ==  typeid( event ) )
  {
    ::itk::ProcessObject::Pointer  process = 
        dynamic_cast< itk::ProcessObject *>( caller );

    const double value = static_cast<double>(process->GetProgress() );
    update_progress( value );
    }

  else if ( typeid( itk::IterationEvent ) == typeid( event ) )
  {
    ::itk::ProcessObject::Pointer  process = 
	dynamic_cast< itk::ProcessObject *>( caller );
    
    update_after_iteration();
  }

}


// Manage a Progress event 
void 
ThresholdSegmentationLevelSetImageFilter::ConstProcessEvent(const itk::Object * caller, const itk::EventObject & event )
{
  if (gui_stop_.get() == 1) return;

  if( typeid( itk::ProgressEvent )   ==  typeid( event ) )
  {
    ::itk::ProcessObject::ConstPointer  process = 
        dynamic_cast< const itk::ProcessObject *>( caller );

    const double value = static_cast<double>(process->GetProgress() );
    update_progress( value );
    }

  else if ( typeid( itk::IterationEvent ) == typeid( event ) )
  {
    ::itk::ProcessObject::ConstPointer  process = 
	dynamic_cast< const itk::ProcessObject *>( caller );
    
    update_after_iteration();
  }

}


void 
ThresholdSegmentationLevelSetImageFilter::update_after_iteration()
{
  if (gui_stop_.get() == 1) return;

  if(gui_update_OutputImage_.get() && iterationCounter_OutputImage%gui_update_iters_OutputImage_.get() == 0 && iterationCounter_OutputImage > 0) {

    // determine type and call do it
    if(0) { } 
   
    else if(do_it_OutputImage< itk::Image<float, 2>, itk::Image<float, 2> >( )) { } 
    else if(do_it_OutputImage< itk::Image<float, 3>, itk::Image<float, 3> >( )) { }
    else {
      // error
      error("Incorrect filter type");
      return;
    }
  }

  // check stop and if true, stop segmentation
  gui_stop_.reset();
  
  if (gui_stop_.get() == 1) {
    // determine type and call do it
    if(0) { } 
    
    else if(do_it_stop< itk::Image<float, 2>, itk::Image<float, 2> >( )) { } 
    else if(do_it_stop< itk::Image<float, 3>, itk::Image<float, 3> >( )) { }
    else {
      // error
      error("Incorrect filter type");
      return;
    }
  }

  iterationCounter_OutputImage++;

}


template<class ImageType, class FeatureImageType>
bool 
ThresholdSegmentationLevelSetImageFilter::do_it_OutputImage()
{
  if (gui_stop_.get() == 1) return true;

  // Move the pixel container and image information of the image 
  // we are working on into a temporary image to  use as the 
  // input to the mini-pipeline.  This avoids a complete copy of the image.

  typedef typename itk::ThresholdSegmentationLevelSetImageFilter< ImageType, FeatureImageType > FilterType;
  
  if(!dynamic_cast<FilterType*>(filter_.GetPointer())) {
    return false;
  }
 
  
  typename FeatureImageType::Pointer tmp = FeatureImageType::New();
  tmp->SetRequestedRegion( dynamic_cast<FilterType*>(filter_.GetPointer())->GetOutput()->GetRequestedRegion() );
  tmp->SetBufferedRegion( dynamic_cast<FilterType*>(filter_.GetPointer())->GetOutput()->GetBufferedRegion() );
  tmp->SetLargestPossibleRegion( dynamic_cast<FilterType*>(filter_.GetPointer())->GetOutput()->GetLargestPossibleRegion() );
  tmp->SetPixelContainer( dynamic_cast<FilterType*>(filter_.GetPointer())->GetOutput()->GetPixelContainer() );
  tmp->CopyInformation( dynamic_cast<FilterType*>(filter_.GetPointer())->GetOutput() );
  
  
  // send segmentation down
  ITKDatatype* out_OutputImage_ = scinew ITKDatatype; 
  out_OutputImage_->data_ = tmp;
  outhandle_OutputImage_ = out_OutputImage_; 
  outport_OutputImage_->send_intermediate(outhandle_OutputImage_);
  return true;
}

template<class ImageType, class FeatureImageType>
bool 
ThresholdSegmentationLevelSetImageFilter::do_it_stop()
{
  if (gui_stop_.get() == 1) return true;

  // stop segmentation
  typedef typename itk::ThresholdSegmentationLevelSetImageFilter< ImageType, FeatureImageType > FilterType;
  
  if(!dynamic_cast<FilterType*>(filter_.GetPointer())) {
    return false;
  }
  
  dynamic_cast<FilterType*>(filter_.GetPointer())->AbortGenerateDataOn();
  return true;
}

// Manage a Progress event 
void 
ThresholdSegmentationLevelSetImageFilter::Observe( itk::Object *caller )
{
  if (gui_stop_.get() == 1) return;

  caller->AddObserver(  itk::ProgressEvent(), m_RedrawCommand.GetPointer() );
  caller->AddObserver(  itk::IterationEvent(), m_RedrawCommand.GetPointer() );
}

void 
ThresholdSegmentationLevelSetImageFilter::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}


} // End of namespace Insight
