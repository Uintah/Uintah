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

#include <Insight/Dataflow/Ports/ITKDatatypePort.h>
#include "itkImageFileWriter.h"
#include "itkCastImageFilter.h"


namespace Insight {

using namespace SCIRun;

class InsightSHARE ImageFileWriter : public Module {
public:

  //! GUI variables
  GuiString gui_FileName_;

  itk::Object::Pointer writer_;
  
  ITKDatatypeIPort* inport_;
  ITKDatatypeHandle inhandle_;

  ImageFileWriter(GuiContext*);

  virtual ~ImageFileWriter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(ImageFileWriter)
ImageFileWriter::ImageFileWriter(GuiContext* ctx)
  : Module("ImageFileWriter", ctx, Source, "DataIO", "Insight"),
    gui_FileName_(ctx->subVar("FileName"))
{
}

ImageFileWriter::~ImageFileWriter(){
}

void ImageFileWriter::execute(){
  // check ports
  inport_ = (ITKDatatypeIPort *)get_iport("Image");
  if(!inport_) {
    error("Unable to initialize iport 'Image'");
  }

  inport_->get(inhandle_);
  if(!inhandle_.get_rep()) {
    return;
  }

  string fn = gui_FileName_.get();

  ///////////////////////
  // <float, 2>
  ///////////////////////
  if(dynamic_cast<itk::Image<float,2>* >(inhandle_.get_rep()->data_.GetPointer())) {
    typedef itk::Image<float, 2> ImageType;
    typedef itk::Image<unsigned short, 2> ShortImageType;
    typedef itk::ImageFileWriter<ShortImageType> FileWriterType;

    writer_ = FileWriterType::New();

    itk::CastImageFilter<ImageType, ShortImageType>::Pointer to_char = itk::CastImageFilter<ImageType, ShortImageType>::New();
    
    itk::Object *object = inhandle_.get_rep()->data_.GetPointer();
    ImageType *img = dynamic_cast<ImageType *>(object);
    
    if( !img ) {
      // error
      return;
    }
    to_char->SetInput(img);
    to_char->Update();
    
    dynamic_cast<FileWriterType *>(writer_.GetPointer())->SetFileName( fn.c_str() );
    dynamic_cast<FileWriterType *>(writer_.GetPointer())->SetInput(to_char->GetOutput());
    
    try
    {
      dynamic_cast<FileWriterType *>(writer_.GetPointer())->Update();
    }
    catch (itk::ExceptionObject &e)
    {
      std::cerr << e << std::endl;
    }
  }
  ///////////////////////
  // <float, 3>
  ///////////////////////
  else if(dynamic_cast<itk::Image<float,3>* >(inhandle_.get_rep()->data_.GetPointer())) {
    typedef itk::Image<float, 3> ImageType;
    typedef itk::Image<unsigned short, 3> ShortImageType;
    typedef itk::ImageFileWriter<ShortImageType> FileWriterType;
    
    writer_ = FileWriterType::New();

    itk::CastImageFilter<ImageType, ShortImageType>::Pointer to_char = itk::CastImageFilter<ImageType, ShortImageType>::New();
    
    itk::Object *object = inhandle_.get_rep()->data_.GetPointer();
    ImageType *img = dynamic_cast<ImageType *>(object);
    
    if( !img ) {
      // error
      return;
    }
    to_char->SetInput(img);
    to_char->Update();
    
    dynamic_cast<FileWriterType *>(writer_.GetPointer())->SetFileName( fn.c_str() );
    dynamic_cast<FileWriterType *>(writer_.GetPointer())->SetInput(to_char->GetOutput());
    
    try
    {
      dynamic_cast<FileWriterType *>(writer_.GetPointer())->Update();
    }
    catch (itk::ExceptionObject &e)
    {
      std::cerr << e << std::endl;
    }
  }
  else 
  {
    // unknown type
    error("Unknown type");
    return;
  }
}

void
 ImageFileWriter::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight


