/*
 * CannySegmentationLevelSetImageFilter.cc
 *
 *   Auto Generated File For CannySegmentationLevelSetImageFilter
 *
 */


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/Insight/share/share.h>

#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>

#include <itkCannySegmentationLevelSetImageFilter.h>

namespace Insight 
{

using namespace SCIRun;

class InsightSHARE CannySegmentationLevelSetImageFilter : public Module 
{
public:

  // Declare GuiVars
  GuiInt gui_iterations_;
  GuiInt gui_negative_features_;
  GuiDouble gui_max_rms_change_;
  GuiDouble gui_threshold_;
  GuiDouble gui_variance_;
  GuiDouble gui_propagation_scaling_;
  GuiDouble gui_advection_scaling_;
  GuiDouble gui_curvature_scaling_;
  GuiDouble gui_isovalue_;
    
  // Declare Ports
  ITKDatatypeIPort* inport1_;
  ITKDatatypeHandle inhandle1_;

  ITKDatatypeIPort* inport2_;
  ITKDatatypeHandle inhandle2_;

  ITKDatatypeOPort* outport1_;
  ITKDatatypeHandle outhandle1_;

  ITKDatatypeOPort* outport2_;
  ITKDatatypeHandle outhandle2_;

  
  CannySegmentationLevelSetImageFilter(GuiContext*);

  virtual ~CannySegmentationLevelSetImageFilter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  // Run function will dynamically cast data to determine which
  // instantiation we are working with. The last template type
  // refers to the last template type of the filter intstantiation.
  template< class InputImageType,  class FeatureImageType,  class OutputPixelType > 
  bool run( itk::Object*, itk::Object* );

};


template<class InputImageType, class FeatureImageType, class OutputPixelType>
bool CannySegmentationLevelSetImageFilter::run( itk::Object *obj1, itk::Object *obj2) 
{
  InputImageType *data1 = dynamic_cast<  InputImageType * >(obj1);
  if( !data1 ) {
    return false;
  }

  FeatureImageType *data2 = dynamic_cast<  FeatureImageType * >(obj2);
  if( !data2 ) {
    return false;
  }


  // create a new filter
  itk::CannySegmentationLevelSetImageFilter< InputImageType, FeatureImageType, OutputPixelType >::Pointer filter = itk::CannySegmentationLevelSetImageFilter< InputImageType, FeatureImageType, OutputPixelType >::New();

  // set filter 
  
  filter->SetMaximumIterations( gui_iterations_.get() );
  
  filter->SetUseNegativeFeatures( gui_negative_features_.get() );
  
  filter->SetMaximumRMSError( gui_max_rms_change_.get() );
  
  filter->SetThreshold( gui_threshold_.get() );
  
  filter->SetVariance( gui_variance_.get() );
  
  filter->SetPropagationScaling( gui_propagation_scaling_.get() );
  
  filter->SetAdvectionScaling( gui_advection_scaling_.get() );
  
  filter->SetCurvatureScaling( gui_curvature_scaling_.get() );
  
  filter->SetIsoSurfaceValue( gui_isovalue_.get() );
     
  // set inputs 

  filter->SetInput( data1 );
   
  filter->SetFeatureImage( data2 );
   

  // execute the filter
  try {

    filter->Update();

  } catch ( itk::ExceptionObject & err ) {
     error("ExceptionObject caught!");
     error(err.GetDescription());
  }

  // get filter output
  
  if(!outhandle1_.get_rep())
  {
    ITKDatatype* im = scinew ITKDatatype;
    im->data_ = filter->GetOutput();
    outhandle1_ = im; 
  }
  
  if(!outhandle2_.get_rep())
  {
    ITKDatatype* im = scinew ITKDatatype;
    im->data_ = filter->GetSpeedImage();
    outhandle2_ = im; 
  }
  
  return true;
}


DECLARE_MAKER(CannySegmentationLevelSetImageFilter)

CannySegmentationLevelSetImageFilter::CannySegmentationLevelSetImageFilter(GuiContext* ctx)
  : Module("CannySegmentationLevelSetImageFilter", ctx, Source, "Filters", "Insight"),
     gui_iterations_(ctx->subVar("iterations")),
     gui_negative_features_(ctx->subVar("negative_features")),
     gui_max_rms_change_(ctx->subVar("max_rms_change")),
     gui_threshold_(ctx->subVar("threshold")),
     gui_variance_(ctx->subVar("variance")),
     gui_propagation_scaling_(ctx->subVar("propagation_scaling")),
     gui_advection_scaling_(ctx->subVar("advection_scaling")),
     gui_curvature_scaling_(ctx->subVar("curvature_scaling")),
     gui_isovalue_(ctx->subVar("isovalue"))
{
}

CannySegmentationLevelSetImageFilter::~CannySegmentationLevelSetImageFilter() 
{
}

void CannySegmentationLevelSetImageFilter::execute() 
{
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
  inport2_ = (ITKDatatypeIPort *)get_iport("FeatureImage");
  if(!inport2_) {
    error("Unable to initialize iport");
    return;
  }

  inport2_->get(inhandle2_);
  if(!inhandle2_.get_rep()) {
    return;
  }

  // check output ports
  outport1_ = (ITKDatatypeOPort *)get_oport("OutputModel");
  if(!outport1_) {
    error("Unable to initialize oport");
    return;
  }
  outport2_ = (ITKDatatypeOPort *)get_oport("OutputFeatures");
  if(!outport2_) {
    error("Unable to initialize oport");
    return;
  }

  // get input
  itk::Object* data1 = inhandle1_.get_rep()->data_.GetPointer();
  itk::Object* data2 = inhandle2_.get_rep()->data_.GetPointer();
  
  // can we operate on it?
  if(0) { } 
  else if(run< itk::Image<unsigned char, 2>, itk::Image<float, 2>, float >( data1, data2 )) { }
  else {
    // error
    error("Incorrect input type");
    return;
  }

  // send the data downstream
  outport1_->send(outhandle1_);
  outport2_->send(outhandle2_);
  
}

void CannySegmentationLevelSetImageFilter::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);

}


} // End of namespace Insight
