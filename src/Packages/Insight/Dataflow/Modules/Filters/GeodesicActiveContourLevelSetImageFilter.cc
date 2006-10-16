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
 * GeodesicActiveContourLevelSetImageFilter.cc
 *
 *
 *
 */


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>

#include <itkGeodesicActiveContourLevelSetImageFilter.h>

#include <itkCommand.h>

namespace Insight
{

using namespace SCIRun;

class GeodesicActiveContourLevelSetImageFilter : public Module
{
public:

  typedef itk::MemberCommand< GeodesicActiveContourLevelSetImageFilter > RedrawCommandType;

  // Filter Declaration
  itk::Object::Pointer filter_;


 // Declare GuiVars
  GuiDouble gui_dervSigma_;
  GuiDouble gui_curvature_scaling_;
  GuiDouble gui_propagation_scaling_;
  GuiDouble gui_advection_scaling_;
  GuiInt gui_max_iterations_;
  GuiDouble gui_max_rms_change_;
  GuiInt gui_reverse_expansion_direction_;
  GuiDouble gui_isovalue_;
  
  GuiInt gui_update_OutputImage_;
  GuiInt gui_update_iters_OutputImage_;
  GuiInt gui_reset_filter_;

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

  
  GeodesicActiveContourLevelSetImageFilter(GuiContext*);

  virtual ~GeodesicActiveContourLevelSetImageFilter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  // Run function will dynamically cast data to determine which
  // instantiation we are working with. The last template type
  // refers to the last template type of the filter intstantiation.
  template<class InputImageType, class FeatureImageType >
  bool run( itk::Object* , itk::Object* );

  // progress bar

  void update_after_iteration();

  void ProcessEvent(itk::Object * caller, const itk::EventObject & event );
  void ConstProcessEvent(const itk::Object * caller, const itk::EventObject & event );
  void Observe( itk::Object *caller );
  RedrawCommandType::Pointer m_RedrawCommand;

  template<class InputImageType, class FeatureImageType>
  bool do_it_OutputImage();
  unsigned int iterationCounter_OutputImage;

};

template<class InputImageType, class FeatureImageType>
bool
GeodesicActiveContourLevelSetImageFilter::run( itk::Object *obj_SeedImage, itk::Object *obj_FeatureImage)
{
  InputImageType *data_SeedImage = dynamic_cast< InputImageType * >(obj_SeedImage);
  
  if( !data_SeedImage ) {
    return false;
  }
  FeatureImageType *data_FeatureImage = dynamic_cast< FeatureImageType * >(obj_FeatureImage);
  
  if( !data_FeatureImage ) {
    return false;
  }

  typedef typename itk::GeodesicActiveContourLevelSetImageFilter< InputImageType, FeatureImageType > FilterType;

  // Check if filter_ has been created
  // or the input data has changed. If
  // this is the case, set the inputs.


  if(gui_reset_filter_.get() == 1 || !filter_ ||
     inhandle_SeedImage_->generation != last_SeedImage_ ||
     inhandle_FeatureImage_->generation != last_FeatureImage_) {

     gui_reset_filter_.set(0);
     
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
   
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetDerivativeSigma( gui_dervSigma_.get() );
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetCurvatureScaling( gui_curvature_scaling_.get() );
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetPropagationScaling( gui_propagation_scaling_.get() );
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetAdvectionScaling( gui_advection_scaling_.get() );
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetNumberOfIterations( gui_max_iterations_.get() );
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetMaximumRMSError( gui_max_rms_change_.get() );

  if( gui_reverse_expansion_direction_.get() ) {
    dynamic_cast<FilterType* >(filter_.GetPointer())->ReverseExpansionDirectionOn( );
  }
  else {
    dynamic_cast<FilterType* >(filter_.GetPointer())->ReverseExpansionDirectionOff( );
  }
  
  dynamic_cast<FilterType* >(filter_.GetPointer())->SetIsoSurfaceValue( gui_isovalue_.get() );

// cerr << "DervSigma " << gui_dervSigma_.get() << endl;
// cerr << "Curv " << gui_curvature_scaling_.get() << endl;
// cerr << "Prop " << gui_propagation_scaling_.get() << endl;
// cerr << "Advec " << gui_advection_scaling_.get() << endl;
// cerr << "Iter " << gui_max_iterations_.get() << endl;
// cerr << "RMS " << gui_max_rms_change_.get() << endl;
// cerr << "RevExpan " << gui_reverse_expansion_direction_.get() << endl;
// cerr << "IsoVal " << gui_isovalue_.get() << endl;
  
  
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
  
   out_SpeedImage_->data_ = const_cast<itk::Image<float,::itk::GetImageDimension<InputImageType>::ImageDimension>* >(dynamic_cast<FilterType* >(filter_.GetPointer())->GetSpeedImage());
  
  outhandle_SpeedImage_ = out_SpeedImage_;
  outport_SpeedImage_->send(outhandle_SpeedImage_);
  

  return true;
}


DECLARE_MAKER(GeodesicActiveContourLevelSetImageFilter)

GeodesicActiveContourLevelSetImageFilter::GeodesicActiveContourLevelSetImageFilter(GuiContext* ctx)
  : Module("GeodesicActiveContourLevelSetImageFilter", ctx, Source, "Filters", "Insight"),
    gui_dervSigma_(get_ctx()->subVar("derivativeSigma")),
     gui_curvature_scaling_(get_ctx()->subVar("curvatureScaling")),
     gui_propagation_scaling_(get_ctx()->subVar("propagationScaling")),
     gui_advection_scaling_(get_ctx()->subVar("advectionScaling")),
     gui_max_iterations_(get_ctx()->subVar("max_iterations")),
     gui_max_rms_change_(get_ctx()->subVar("max_rms_change")),
     gui_reverse_expansion_direction_(get_ctx()->subVar("reverse_expansion_direction")),
     gui_isovalue_(get_ctx()->subVar("isovalue")),
     gui_update_OutputImage_(get_ctx()->subVar("update_OutputImage")),
     gui_update_iters_OutputImage_(get_ctx()->subVar("update_iters_OutputImage")),
     gui_reset_filter_(get_ctx()->subVar("reset_filter")),
     last_SeedImage_(-1),
    last_FeatureImage_(-1)
{
  filter_ = 0;

  m_RedrawCommand = RedrawCommandType::New();
  m_RedrawCommand->SetCallbackFunction( this, &GeodesicActiveContourLevelSetImageFilter::ProcessEvent );
  m_RedrawCommand->SetCallbackFunction( this, &GeodesicActiveContourLevelSetImageFilter::ConstProcessEvent );

  iterationCounter_OutputImage = 0;

  update_progress(0.0);

}

GeodesicActiveContourLevelSetImageFilter::~GeodesicActiveContourLevelSetImageFilter()
{
}

void
GeodesicActiveContourLevelSetImageFilter::execute()
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

  iterationCounter_OutputImage = 0;
  gui_update_OutputImage_.reset();
  gui_update_iters_OutputImage_.reset();

  // get input
  itk::Object* data_SeedImage = inhandle_SeedImage_.get_rep()->data_.GetPointer();
  itk::Object* data_FeatureImage = inhandle_FeatureImage_.get_rep()->data_.GetPointer();
    
  // can we operate on it?
  if(0) { }
  else if(run< itk::Image<float, 2>, itk::Image<float, 2> >( data_SeedImage, data_FeatureImage )) { }
  else if(run< itk::Image<float, 3>, itk::Image<float, 3> >( data_SeedImage, data_FeatureImage )) { }
  else if(run< itk::Image<double, 2>, itk::Image<double, 2> >( data_SeedImage, data_FeatureImage )) { }
  else if(run< itk::Image<double, 3>, itk::Image<double, 3> >( data_SeedImage, data_FeatureImage )) { }
  else {
    // error
    error("Incorrect input type");
    return;
  }

}

// Manage a Progress event
void
GeodesicActiveContourLevelSetImageFilter::ProcessEvent( itk::Object * caller, const itk::EventObject & event )
{
  if( typeid( itk::ProgressEvent ) == typeid( event ) )
  {
    ::itk::ProcessObject::Pointer process =
        dynamic_cast< itk::ProcessObject *>( caller );

    const double value = static_cast<double>(process->GetProgress() );
    update_progress( value );
    }

  else if ( typeid( itk::IterationEvent ) == typeid( event ) )
  {
    ::itk::ProcessObject::Pointer process =
dynamic_cast< itk::ProcessObject *>( caller );
    
    update_after_iteration();
  }

}


// Manage a Progress event
void
GeodesicActiveContourLevelSetImageFilter::ConstProcessEvent(const itk::Object * caller, const itk::EventObject & event )
{
  if( typeid( itk::ProgressEvent ) == typeid( event ) )
  {
    ::itk::ProcessObject::ConstPointer process =
        dynamic_cast< const itk::ProcessObject *>( caller );

    const double value = static_cast<double>(process->GetProgress() );
    update_progress( value );
    }

  else if ( typeid( itk::IterationEvent ) == typeid( event ) )
  {
    ::itk::ProcessObject::ConstPointer process =
dynamic_cast< const itk::ProcessObject *>( caller );
    
    update_after_iteration();
  }

}


void
GeodesicActiveContourLevelSetImageFilter::update_after_iteration()
{

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
  iterationCounter_OutputImage++;

}


template<class InputImageType, class FeatureImageType>
bool
GeodesicActiveContourLevelSetImageFilter::do_it_OutputImage()
{
  // Move the pixel container and image information of the image
  // we are working on into a temporary image to use as the
  // input to the mini-pipeline. This avoids a complete copy of the image.

  typedef typename itk::GeodesicActiveContourLevelSetImageFilter< InputImageType, FeatureImageType > FilterType;
  
  if(!dynamic_cast<FilterType*>(filter_.GetPointer())) {
    return false;
  }
 
  
  typename itk::Image<float,::itk::GetImageDimension<InputImageType>::ImageDimension>::Pointer tmp = itk::Image<float,::itk::GetImageDimension<InputImageType>::ImageDimension>::New();

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

// Manage a Progress event
void
GeodesicActiveContourLevelSetImageFilter::Observe( itk::Object *caller )
{
  caller->AddObserver( itk::ProgressEvent(), m_RedrawCommand.GetPointer() );
  caller->AddObserver( itk::IterationEvent(), m_RedrawCommand.GetPointer() );
}

void
GeodesicActiveContourLevelSetImageFilter::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("GeodesicActiveContourLevelSetImageFilter needs a minor command");
    return;
  }

  if (args[1] == "stop_segmentation") {
    // since we only support float images in 2 and 3 dimensions, we 
    // only have 2 cases to check for
    typedef itk::GeodesicActiveContourLevelSetImageFilter< itk::Image<float,2>, itk::Image<float, 2> > FilterType2D;
    typedef itk::GeodesicActiveContourLevelSetImageFilter< itk::Image<float,3>, itk::Image<float, 3> > FilterType3D;
    if (dynamic_cast<FilterType2D *>(filter_.GetPointer())) {
      dynamic_cast<FilterType2D *>(filter_.GetPointer())->AbortGenerateDataOn();
    } else if (dynamic_cast<FilterType3D *>(filter_.GetPointer())) {
      dynamic_cast<FilterType3D *>(filter_.GetPointer())->AbortGenerateDataOn();
    }
  } else {
    Module::tcl_command(args, userdata);
  }

}
} // End of namespace Insight
