/* 
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#include "itkVector.h"


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

  itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);

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
    gui_FileName_(ctx->subVar("filename"))
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
  else if(run<itk::Image<double,2> >( data )) { } 
  else if(run<itk::Image<double,3> >( data )) { } 
  else if(run<itk::Image<unsigned char,2> >( data )) { } 
  else if(run<itk::Image<unsigned char,3> >( data )) { } 
  else if(run<itk::Image<unsigned short,2> >( data )) {  }
  else if(run<itk::Image<unsigned short,3> >( data )) { } 
  else if(run<itk::Image<unsigned long,2> >( data )) { } 
  else if(run<itk::Image<unsigned long,3> >( data )) { } 
  else if(run<itk::Image<itk::RGBPixel<unsigned char>,2> >( data )) { } 
  else if(run<itk::Image<itk::RGBPixel<unsigned char>,3> >( data )) { } 
  else if(run<itk::Image<itk::Vector<float>,2> >( data )) { } 
  else if(run<itk::Image<itk::Vector<float>,3> >( data )) { } 
  else if(run<itk::Image<itk::Vector<double>,2> >( data )) { } 
  else if(run<itk::Image<itk::Vector<double>,3> >( data )) { } 
  else if(run<itk::Image<itk::Vector<unsigned char>,2> >( data )) { } 
  else if(run<itk::Image<itk::Vector<unsigned char>,2> >( data )) { } 
  else if(run<itk::Image<itk::Vector<unsigned short>,2> >( data )) { }
  else if(run<itk::Image<itk::Vector<unsigned short>,3> >( data )) { } 
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


