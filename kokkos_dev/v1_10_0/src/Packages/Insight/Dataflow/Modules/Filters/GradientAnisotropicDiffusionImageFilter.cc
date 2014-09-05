/*
 * GradientAnisotropicDiffusionImageFilter.cc
 *
 *   Auto Generated File For GradientAnisotropicDiffusionImageFilter
 *
 */


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/Insight/share/share.h>

#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>

#include <itkGradientAnisotropicDiffusionImageFilter.h>

namespace Insight 
{

using namespace SCIRun;

class InsightSHARE GradientAnisotropicDiffusionImageFilter : public Module 
{
public:

  // Declare GuiVars
  GuiDouble gui_time_step_;
  GuiInt gui_iterations_;
  GuiDouble gui_conductance_parameter_;
    
  // Declare Ports
  ITKDatatypeIPort* inport1_;
  ITKDatatypeHandle inhandle1_;

  ITKDatatypeOPort* outport1_;
  ITKDatatypeHandle outhandle1_;

  
  GradientAnisotropicDiffusionImageFilter(GuiContext*);

  virtual ~GradientAnisotropicDiffusionImageFilter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  // Run function will dynamically cast data to determine which
  // instantiation we are working with. The last template type
  // refers to the last template type of the filter intstantiation.
  template< class InputImageType,  class OutputImageType > 
  bool run( itk::Object* );

};


template<class InputImageType, class OutputImageType>
bool GradientAnisotropicDiffusionImageFilter::run( itk::Object *obj1) 
{
  InputImageType *data1 = dynamic_cast<  InputImageType * >(obj1);
  if( !data1 ) {
    return false;
  }


  // create a new filter
  itk::GradientAnisotropicDiffusionImageFilter< InputImageType, OutputImageType >::Pointer filter = itk::GradientAnisotropicDiffusionImageFilter< InputImageType, OutputImageType >::New();

  // set filter 
  
  filter->SetTimeStep( gui_time_step_.get() );
  
  filter->SetIterations( gui_iterations_.get() );
  
  filter->SetConductanceParameter( gui_conductance_parameter_.get() );
     
  // set inputs 

  filter->SetInput( data1 );
   

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
  
  return true;
}


DECLARE_MAKER(GradientAnisotropicDiffusionImageFilter)

GradientAnisotropicDiffusionImageFilter::GradientAnisotropicDiffusionImageFilter(GuiContext* ctx)
  : Module("GradientAnisotropicDiffusionImageFilter", ctx, Source, "Filters", "Insight"),
     gui_time_step_(ctx->subVar("time_step")),
     gui_iterations_(ctx->subVar("iterations")),
     gui_conductance_parameter_(ctx->subVar("conductance_parameter"))
{
}

GradientAnisotropicDiffusionImageFilter::~GradientAnisotropicDiffusionImageFilter() 
{
}

void GradientAnisotropicDiffusionImageFilter::execute() 
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

  // check output ports
  outport1_ = (ITKDatatypeOPort *)get_oport("OutputImage");
  if(!outport1_) {
    error("Unable to initialize oport");
    return;
  }

  // get input
  itk::Object* data1 = inhandle1_.get_rep()->data_.GetPointer();
  
  // can we operate on it?
  if(0) { } 
  else if(run< itk::Image<float, 2>, itk::Image<float, 2 > >( data1 )) { } 
  else if(run< itk::Image<float, 3>, itk::Image<float, 3 > >( data1 )) { }
  else {
    // error
    error("Incorrect input type");
    return;
  }

  // send the data downstream
  outport1_->send(outhandle1_);
  
}

void GradientAnisotropicDiffusionImageFilter::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);

}


} // End of namespace Insight
