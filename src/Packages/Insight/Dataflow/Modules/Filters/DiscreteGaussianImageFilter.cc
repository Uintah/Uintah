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

#include <Insight/Dataflow/Ports/ITKImagePort.h>

//itk includes
#include "itkDiscreteGaussianImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkPNGImageIO.h"


namespace Insight {

using namespace SCIRun;

  typedef itk::DiscreteGaussianImageFilter<ImageType, ImageType> FilterType;
  typedef itk::PNGImageIO IOType;

class InsightSHARE DiscreteGaussianImageFilter : public Module {
public:
  //! GUI variables
  GuiDouble gui_variance_;
  GuiDouble gui_max_error_;

  FilterType::Pointer filter_;

  ITKImageIPort* inport_;
  ITKImageHandle inhandle_;
  ITKImageOPort* outport_;
  ITKImageHandle outhandle_;

  DiscreteGaussianImageFilter(GuiContext*);

  virtual ~DiscreteGaussianImageFilter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(DiscreteGaussianImageFilter)
DiscreteGaussianImageFilter::DiscreteGaussianImageFilter(GuiContext* ctx)
  : Module("DiscreteGaussianImageFilter", ctx, Source, "Filters", "Insight"),
    gui_variance_(ctx->subVar("variance")), 
    gui_max_error_(ctx->subVar("max_error"))
{
  filter_  = FilterType::New();
}

DiscreteGaussianImageFilter::~DiscreteGaussianImageFilter(){
}

void
 DiscreteGaussianImageFilter::execute(){

  inport_ = (ITKImageIPort *)get_iport("Image");
  if(!inport_) {
    error("Unable to initialize iport 'ITKImage'");
    return;
  }

  inport_->get(inhandle_);
  if(!inhandle_.get_rep()) {
    return;
  }
  
  outport_ = (ITKImageOPort *)get_oport("Image");
  if(!outport_) {
    error("Unable to initialize oport 'ITKImage'");
    return;
  }

  double variance = gui_variance_.get();
  double max_error = gui_max_error_.get();

  filter_->SetVariance(variance);
  filter_->SetMaximumError(max_error);

  filter_->SetInput(inhandle_->to_float_->GetOutput());

  if(!outhandle_.get_rep())
  {
    ITKImage *image = scinew ITKImage;
    image->to_short_->SetInput(filter_->GetOutput());
    outhandle_ = image;
  }

  // Send the data downstream
  outport_->send(outhandle_);
  
}

void
 DiscreteGaussianImageFilter::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight


