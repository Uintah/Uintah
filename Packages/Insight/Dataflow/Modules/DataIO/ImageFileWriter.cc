/*
 *  ImageFileWriter.cc:
 *
 *  Written by:
 *   darbyb
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/Insight/share/share.h>

#include <Insight/Dataflow/Ports/ITKImagePort.h>
#include "itkCastImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkPNGImageIO.h"

namespace Insight {

using namespace SCIRun;

  typedef itk::ImageFileWriter<ShortImageType> WriterType;
  typedef itk::PNGImageIO IOType;

class InsightSHARE ImageFileWriter : public Module {
public:

  //! GUI variables
  GuiString gui_filename_;

  WriterType::Pointer writer_;
  IOType::Pointer io_;
  
  ITKImageIPort* inport_;
  ITKImageHandle inhandle_;

  ImageFileWriter(GuiContext*);

  virtual ~ImageFileWriter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(ImageFileWriter)
ImageFileWriter::ImageFileWriter(GuiContext* ctx)
  : Module("ImageFileWriter", ctx, Source, "DataIO", "Insight"),
    gui_filename_(ctx->subVar("filename"))
{
  writer_ = WriterType::New();
  io_ = IOType::New();
}

ImageFileWriter::~ImageFileWriter(){
}

void
 ImageFileWriter::execute(){
  inport_ = (ITKImageIPort *)get_iport("Image");
  if(!inport_) {
    error("Unable to initialize iport 'Image'");
  }

  inport_->get(inhandle_);
  if(!inhandle_.get_rep()) {
    return;
  }

  string fn = gui_filename_.get();

  writer_->SetFileName( fn.c_str() );
  writer_->SetImageIO( io_ );
  
  writer_->SetInput(inhandle_->to_short_->GetOutput());

  try
    {
      writer_->Write();
    }
  catch (itk::ExceptionObject &e)
    {
      std::cerr << e << std::endl;
    }
  }

void
 ImageFileWriter::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight


