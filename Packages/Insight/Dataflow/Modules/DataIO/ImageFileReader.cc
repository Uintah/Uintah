/*
 *  ImageFileReader.cc:
 *
 *  Written by:
 *   darbyb
 *   January 2003
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/Insight/share/share.h>

#include <Insight/Dataflow/Ports/ITKDatatypePort.h>
#include "itkImageFileReader.h"


namespace Insight {

using namespace SCIRun;

class InsightSHARE ImageFileReader : public Module {
public:

  //! GUI variables
  GuiString gui_FileName_;

  ITKDatatypeOPort* outport_;
  ITKDatatypeHandle handle_;
  
  ImageFileReader(GuiContext*);

  virtual ~ImageFileReader();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};

DECLARE_MAKER(ImageFileReader)

ImageFileReader::ImageFileReader(GuiContext* ctx)
  : Module("ImageFileReader", ctx, Source, "DataIO", "Insight"),
    gui_FileName_(ctx->subVar("FileName"))
{
}


ImageFileReader::~ImageFileReader(){
}


void ImageFileReader::execute(){
  // check ports
  outport_ = (ITKDatatypeOPort *)get_oport("Image");
  if(!outport_) {
    error("Unable to initialize oport 'Image'.");
    return;
  }

  // Can't determine image type by casting??
  if(1)
  {
    typedef itk::ImageFileReader<itk::Image<float, 3> > FileReaderType;
    
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


void ImageFileReader::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight


