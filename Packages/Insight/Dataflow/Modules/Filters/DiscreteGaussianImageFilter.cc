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

  ITKDatatypeIPort* inport_;
  ITKDatatypeHandle inhandle_;

  ITKDatatypeOPort* outport_;
  ITKDatatypeHandle outhandle_;

  DiscreteGaussianImageFilter(GuiContext*);

  virtual ~DiscreteGaussianImageFilter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  
  template<class T> bool run(itk::Object* );
};

template<class T>
bool DiscreteGaussianImageFilter::run( itk::Object *obj )
{
  T *data = dynamic_cast<T * >(obj);
  if ( !data ) {
    return false;
  }

  typedef T ImageType;
  typedef itk::DiscreteGaussianImageFilter<ImageType, ImageType> FilterType;

  // create a new filter
  FilterType::Pointer filter =  FilterType::New();

  // set filter
  filter->SetVariance( gui_Variance_.get() );
  filter->SetMaximumError( gui_MaximumError_.get() );

  filter->SetInput( data ); 

  // run the filter
  filter->Update();

  // get filter output
  if(!outhandle_.get_rep())
  {
    ITKDatatype* im = scinew ITKDatatype;
    im->data_ = filter->GetOutput();
    outhandle_ = im;
  }

  return true;
}


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

  // get input
  itk::Object * data = inhandle_.get_rep()->data_.GetPointer();

  // can we operate on it ?
  if ( !run<itk::Image<float,2> >( data )
        && 
       !run<itk::Image<float,3> >( data )
       )
  {
    // error 
    error("Unknown type");
    return;
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


