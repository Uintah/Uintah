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

  ITKDatatypeIPort* inport_;
  ITKDatatypeHandle inhandle_;

  ImageFileWriter(GuiContext*);

  virtual ~ImageFileWriter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  template<class T1, class T2> bool run(itk::Object* );
};

template<class T1, class T2>
bool ImageFileWriter::run( itk::Object *obj )
{
  T1 *data1 = dynamic_cast<T1 * >(obj);

  if ( !data1 ) {
    return false;
  }

  // create a new writer
  itk::ImageFileWriter<T2>::Pointer writer = itk::ImageFileWriter<T2>::New();
  
  // cast it to a writable type
  itk::CastImageFilter<T1, T2>::Pointer to_char = itk::CastImageFilter<T1, T2>::New();
  
  to_char->SetInput( data1 );
  to_char->Update();

  // set writer
  string fn = gui_FileName_.get();    
  writer->SetFileName( fn.c_str() );
  writer->SetInput(to_char->GetOutput());
  
  // write
  try
    {
      writer->Update();
    }
  catch (itk::ExceptionObject &e)
    {
      std::cerr << e << std::endl;
    }
  return true;
}

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


  // get input
  itk::Object* data = inhandle_.get_rep()->data_.GetPointer();

  // can we operate on it?
  if ( !run<itk::Image<float,2>, itk::Image<unsigned short,2> >( data )
       &&
       !run<itk::Image<float,3>, itk::Image<unsigned short,3> >( data )
       )
    {
          // error 
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


