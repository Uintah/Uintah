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
 *  ImageReaderUShort2D.cc:
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

class InsightSHARE ImageReaderUShort2D : public Module {
public:
  //! GUI variables
  GuiString gui_filename_;

  ITKDatatypeOPort* outport_;
  ITKDatatypeHandle handle_;

  string prevFile;

  ImageReaderUShort2D(GuiContext*);

  virtual ~ImageReaderUShort2D();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(ImageReaderUShort2D)
ImageReaderUShort2D::ImageReaderUShort2D(GuiContext* ctx)
  : Module("ImageReaderUShort2D", ctx, Source, "DataIO", "Insight"),
    gui_filename_(ctx->subVar("filename"))
{
  prevFile = "";
}

ImageReaderUShort2D::~ImageReaderUShort2D(){
}

void ImageReaderUShort2D::execute(){
  // check ports
  outport_ = (ITKDatatypeOPort *)get_oport("OutputImage");
  if(!outport_) {
    error("Unable to initialize oport 'OutputImage'.");
    return;
  }

  typedef itk::ImageFileReader<itk::Image<unsigned short, 2> > FileReaderType;
  
  // create a new reader
  FileReaderType::Pointer reader = FileReaderType::New();
  
  // set reader
  string fn = gui_filename_.get();
  reader->SetFileName( fn.c_str() );
  
  reader->Update();  
  
  // get reader output
  if(!handle_.get_rep() || (fn != prevFile))
    {
      ITKDatatype *im = scinew ITKDatatype;
      im->data_ = reader->GetOutput();  
      handle_ = im; 
      prevFile = fn;
    }
  
  // Send the data downstream
  outport_->send(handle_);
}

void ImageReaderUShort2D::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight


