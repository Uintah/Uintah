/*
 *  UChar2DToFloat2D.cc:
 *
 *  Written by:
 *   darbyb
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/Insight/share/share.h>

#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>

#include <itkCastImageFilter.h>

namespace Insight {

using namespace SCIRun;

class InsightSHARE UChar2DToFloat2D : public Module {
  typedef itk::CastImageFilter< itk::Image<unsigned char,2>, itk::Image<float,2> > CastImageType;
public:

  CastImageType::Pointer to_float_;

  // Declare Ports
  ITKDatatypeIPort* inport1_;
  ITKDatatypeHandle inhandle1_;

  ITKDatatypeOPort* outport1_;
  ITKDatatypeHandle outhandle1_;

  UChar2DToFloat2D(GuiContext*);

  virtual ~UChar2DToFloat2D();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(UChar2DToFloat2D)
UChar2DToFloat2D::UChar2DToFloat2D(GuiContext* ctx)
  : Module("UChar2DToFloat2D", ctx, Source, "Filters", "Insight")
{
  to_float_ = CastImageType::New(); 
}

UChar2DToFloat2D::~UChar2DToFloat2D(){
}

void
 UChar2DToFloat2D::execute(){
  // check input ports
  inport1_ = (ITKDatatypeIPort *)get_iport("Image");
  if(!inport1_) {
    error("Unable to initialize iport");
    return;
  }

  inport1_->get(inhandle1_);
  if(!inhandle1_.get_rep()) {
    return;
  }

  // check output ports
  outport1_ = (ITKDatatypeOPort *)get_oport("Image");
  if(!outport1_) {
    error("Unable to initialize oport");
    return;
  }

  to_float_->SetInput(dynamic_cast<itk::Image<unsigned char, 2>* >(inhandle1_.get_rep()->data_.GetPointer()));
  
  try {
    to_float_->Update();
  } catch ( itk::ExceptionObject & err ) {
     std::cerr << "ExceptionObject caught!" << std::endl;
     std::cerr << err << std::endl;
  }
  
  if(!outhandle1_.get_rep())
  {
    ITKDatatype* im = scinew ITKDatatype;
    im->data_ = to_float_->GetOutput();
    outhandle1_ = im; 
  }
  
  // send the data downstream
  outport1_->send(outhandle1_);
}

void
 UChar2DToFloat2D::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight


