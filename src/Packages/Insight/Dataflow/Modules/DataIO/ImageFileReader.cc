/*
 *  ImageFileReader.cc:
 *
 *  Written by:
 *   darbyb
 *   TODAY'S DATE HERE
 *
 */

#include "ImageFileReader.h"

//itk
#include "itkCastImageFilter.h"
#include "itkImageFileReader.h"
#include "itkPNGImageIO.h"

namespace Insight {

using namespace SCIRun;

DECLARE_MAKER(ImageFileReader)
ImageFileReader::ImageFileReader(GuiContext* ctx)
  : Module("ImageFileReader", ctx, Source, "DataIO", "Insight"),
    gui_filename_(ctx->subVar("filename"))
{
  reader_ = ReaderType::New();
  io_ = IOType::New();
}

ImageFileReader::~ImageFileReader(){
}

void
 ImageFileReader::execute(){

  outport_ = (ITKImageOPort *)get_oport("Image");
  if(!outport_) {
    error("Unable to initialize oport 'Image'.");
    return;
  }

  string fn = gui_filename_.get();

  reader_->SetFileName( fn.c_str() );
  reader_->SetImageIO(io_);

  if(!handle_.get_rep())
  {
    ITKImage *image = scinew ITKImage;
    image->to_float_->SetInput(reader_->GetOutput());
    handle_ = image;
  }

  // Send the data downstream
  outport_->send(handle_);
}

void
 ImageFileReader::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight


