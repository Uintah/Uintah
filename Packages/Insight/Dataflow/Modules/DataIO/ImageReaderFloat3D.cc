/*
 *  ImageReaderFloat3D.cc:
 *
 *  Written by:
 *   darbyb
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/Insight/share/share.h>

#include <Insight/Dataflow/Ports/ITKDatatypePort.h>
#include "itkImageFileReader.h"


namespace Insight {

using namespace SCIRun;

class InsightSHARE ImageReaderFloat3D : public Module {
public:
  //! GUI variables
  GuiString gui_filename_;

  ITKDatatypeOPort* outport_;
  ITKDatatypeHandle handle_;

  ImageReaderFloat3D(GuiContext*);

  virtual ~ImageReaderFloat3D();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(ImageReaderFloat3D)
ImageReaderFloat3D::ImageReaderFloat3D(GuiContext* ctx)
  : Module("ImageReaderFloat3D", ctx, Source, "DataIO", "Insight"),
    gui_filename_(ctx->subVar("filename"))
{
}

ImageReaderFloat3D::~ImageReaderFloat3D(){
}

void
 ImageReaderFloat3D::execute(){
  // check ports
  outport_ = (ITKDatatypeOPort *)get_oport("Image1");
  if(!outport_) {
    error("Unable to initialize oport 'Image1'.");
    return;
  }

  typedef itk::ImageFileReader<itk::Image<float, 3> > FileReaderType;
  
  // create a new reader
  FileReaderType::Pointer reader = FileReaderType::New();
  
  // set reader
  string fn = gui_filename_.get();
  reader->SetFileName( fn.c_str() );
  
  reader->Update();  
  
  // get reader output
  if(!handle_.get_rep())
    {
      ITKDatatype *im = scinew ITKDatatype;
      im->data_ = reader->GetOutput();  
      handle_ = im; 
    }
  
  // Send the data downstream
  outport_->send(handle_);
}

void
 ImageReaderFloat3D::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight


