/*
 *  ImageReaderUChar2D.cc:
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

class InsightSHARE ImageReaderUChar2D : public Module {
public:

  //! GUI variables
  GuiString gui_FileName_;

  ITKDatatypeOPort* outport_;
  ITKDatatypeHandle handle_;
  
  ImageReaderUChar2D(GuiContext*);

  virtual ~ImageReaderUChar2D();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(ImageReaderUChar2D)
ImageReaderUChar2D::ImageReaderUChar2D(GuiContext* ctx)
  : Module("ImageReaderUChar2D", ctx, Source, "DataIO", "Insight"),
    gui_FileName_(ctx->subVar("FileName"))
{
}

ImageReaderUChar2D::~ImageReaderUChar2D(){
}

void
 ImageReaderUChar2D::execute(){
  // check ports
  outport_ = (ITKDatatypeOPort *)get_oport("Image");
  if(!outport_) {
    error("Unable to initialize oport 'Image'.");
    return;
  }

  // Can't determine image type by casting??
  if(1)
  {
    typedef itk::ImageFileReader<itk::Image<unsigned char, 2> > FileReaderType;
    
    // create a new reader
    FileReaderType::Pointer reader = FileReaderType::New();

    // set reader
    string fn = gui_FileName_.get();
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
  else {
    // unknown type
    error("Unknown type");
    return;
  }
}

void
 ImageReaderUChar2D::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight


