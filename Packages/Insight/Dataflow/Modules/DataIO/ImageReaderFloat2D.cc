/*
 *  ImageReaderFloat2D.cc:
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

class InsightSHARE ImageReaderFloat2D : public Module {
public:
  //! GUI variables
  GuiString gui_filename_;

  ITKDatatypeOPort* outport_;
  ITKDatatypeHandle handle_;

  ImageReaderFloat2D(GuiContext*);

  virtual ~ImageReaderFloat2D();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(ImageReaderFloat2D)
ImageReaderFloat2D::ImageReaderFloat2D(GuiContext* ctx)
  : Module("ImageReaderFloat2D", ctx, Source, "DataIO", "Insight"),
    gui_filename_(ctx->subVar("filename"))
{
}

ImageReaderFloat2D::~ImageReaderFloat2D(){
}

void
 ImageReaderFloat2D::execute(){
  // check ports
  outport_ = (ITKDatatypeOPort *)get_oport("Image1");
  if(!outport_) {
    error("Unable to initialize oport 'Image1'.");
    return;
  }

  typedef itk::ImageFileReader<itk::Image<float, 2> > FileReaderType;
  
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
 ImageReaderFloat2D::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight


