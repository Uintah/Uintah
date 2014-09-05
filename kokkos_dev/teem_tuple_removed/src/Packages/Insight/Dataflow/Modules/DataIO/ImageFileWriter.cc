/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

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

#include "itkRGBPixel.h"


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

  template<class T1> bool run(itk::Object* );
};

template<class T1>
bool ImageFileWriter::run( itk::Object *obj )
{
  T1 *data1 = dynamic_cast<T1 * >(obj);

  if ( !data1 ) {
    return false;
  }

  // create a new writer
  typename itk::ImageFileWriter<T1>::Pointer writer = itk::ImageFileWriter<T1>::New();
  
  // set writer
  string fn = gui_FileName_.get();    
  writer->SetFileName( fn.c_str() );
  writer->SetInput( data1 );
  
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
  if(0) { }
  else if(run<itk::Image<float,2> >( data )) { } 
  else if(run<itk::Image<float,3> >( data )) { } 
  else if(run<itk::Image<unsigned char,2> >( data )) { } 
  else if(run<itk::Image<unsigned char,3> >( data )) { } 
  else if(run<itk::Image<unsigned short,2> >( data )) {  }
  else if(run<itk::Image<unsigned short,3> >( data )) { } 
  else if(run<itk::Image<unsigned long,2> >( data )) { } 
  else if(run<itk::Image<unsigned long,3> >( data )) { } 
  else if(run<itk::Image<itk::RGBPixel<unsigned char>,2> >( data )) { } 
  else if(run<itk::Image<itk::RGBPixel<unsigned char>,3> >( data )) { } 
  else {
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


