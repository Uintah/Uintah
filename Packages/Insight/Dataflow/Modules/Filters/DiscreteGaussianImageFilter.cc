/*
 *  DiscreteGaussianImageFilter.cc:
 *
 *  Written by:
 *   darbyb
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/Insight/share/share.h>
#include <Core/GuiInterface/GuiVar.h>

#include <Insight/Dataflow/Ports/ITKDatatypePort.h>

//itk includes
#include "itkDiscreteGaussianImageFilter.h"

namespace Insight {

using namespace SCIRun;

class InsightSHARE DiscreteGaussianImageFilter : public Module {
public:
  //! GUI variables
  GuiDouble gui_Variance_;
  GuiDouble gui_MaximumError_;

  itk::Object::Pointer filter_;

  ITKDatatypeIPort* inport_;
  ITKDatatypeHandle inhandle_;
  ITKDatatypeOPort* outport_;
  ITKDatatypeHandle outhandle_;

  DiscreteGaussianImageFilter(GuiContext*);

  virtual ~DiscreteGaussianImageFilter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  
};


DECLARE_MAKER(DiscreteGaussianImageFilter)
DiscreteGaussianImageFilter::DiscreteGaussianImageFilter(GuiContext* ctx)
  : Module("DiscreteGaussianImageFilter", ctx, Source, "Filters", "Insight"),
    gui_Variance_(ctx->subVar("Variance")), 
    gui_MaximumError_(ctx->subVar("MaximumError"))
{
}

DiscreteGaussianImageFilter::~DiscreteGaussianImageFilter(){
}


void DiscreteGaussianImageFilter::execute(){
  // check ports
  inport_ = (ITKDatatypeIPort *)get_iport("Image");
  if(!inport_) {
    error("Unable to initialize iport 'ITKDatatype'");
    return;
  }

  inport_->get(inhandle_);
  if(!inhandle_.get_rep()) {
    return;
  }
  
  outport_ = (ITKDatatypeOPort *)get_oport("Image");
  if(!outport_) {
    error("Unable to initialize oport 'ITKDatatype'");
    return;
  }

  // determine filter type

  /////////////////////////////
  // <float, 2>
  ////////////////////////////
  if(dynamic_cast<itk::Image<float,2>* >(inhandle_.get_rep()->data_.GetPointer())) {
    typedef itk::Image<float, 2> ImageType;
    typedef itk::DiscreteGaussianImageFilter<ImageType, ImageType> FilterType;
    
    filter_ =  FilterType::New();
    
    dynamic_cast<FilterType*>(filter_.GetPointer())->SetVariance( gui_Variance_.get() );
    
    dynamic_cast<FilterType*>(filter_.GetPointer())->SetMaximumError( gui_MaximumError_.get() );
    
    
    itk::Object *object = inhandle_.get_rep()->data_.GetPointer();
    ImageType *img = dynamic_cast<ImageType *>(object);
    
    if ( !img ) {
      // error
      return;
    }
    
    dynamic_cast<FilterType*>(filter_.GetPointer())->SetInput( img ); 
    
    dynamic_cast<FilterType*>(filter_.GetPointer())->Update();
    
    if(!outhandle_.get_rep())
    {
      ITKDatatype* im = scinew ITKDatatype;
      im->data_ = dynamic_cast<FilterType*>(filter_.GetPointer())->GetOutput();
      outhandle_ = im;
    }
    
    // Send the data downstream
    outport_->send(outhandle_);
    
  }
  /////////////////////////////
  // <float, 3>
  ////////////////////////////
  else if(dynamic_cast<itk::Image<float,3>* >(inhandle_.get_rep()->data_.GetPointer())) {
    typedef itk::Image<float, 3> ImageType;
    typedef itk::DiscreteGaussianImageFilter<ImageType, ImageType> FilterType;
    
    filter_ =  FilterType::New();
    
    dynamic_cast<FilterType*>(filter_.GetPointer())->SetVariance( gui_Variance_.get() );
    
    dynamic_cast<FilterType*>(filter_.GetPointer())->SetMaximumError( gui_MaximumError_.get() );
    
    itk::Object *object = inhandle_.get_rep()->data_.GetPointer();
    ImageType *img = dynamic_cast<ImageType *>(object);

    if ( !img ) {
      // error
      return;
    }

    dynamic_cast<FilterType*>(filter_.GetPointer())->SetInput( img ); 
    
    dynamic_cast<FilterType*>(filter_.GetPointer())->Update();

    if(!outhandle_.get_rep())
    {
      ITKDatatype* im = scinew ITKDatatype;
      im->data_ = dynamic_cast<FilterType*>(filter_.GetPointer())->GetOutput();
      outhandle_ = im;
    }
    
    // Send the data downstream
    outport_->send(outhandle_);
    
  }
  else {
    // unknown type
    error("Unknown type");
    return;
  }
}

void
 DiscreteGaussianImageFilter::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight


