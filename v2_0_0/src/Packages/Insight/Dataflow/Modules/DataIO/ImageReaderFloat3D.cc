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
 *  ImageReaderFloat3D.cc:
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

class InsightSHARE ImageReaderFloat3D : public Module {
public:
  //! GUI variables
  GuiString gui_filename_;

  ITKDatatypeOPort* outport_;
  ITKDatatypeHandle handle_;

  string prevFile;

  ImageReaderFloat3D(GuiContext*);

  virtual ~ImageReaderFloat3D();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(ImageReaderFloat3D)
ImageReaderFloat3D::ImageReaderFloat3D(GuiContext* ctx)
  : Module("ImageReaderFloat3D", ctx, Source, "DataIO", "Insight"),
    gui_filename_(ctx->subVar("filename"))
{
  prevFile = "";
}

ImageReaderFloat3D::~ImageReaderFloat3D(){
}

void
 ImageReaderFloat3D::execute(){
  // check ports
  outport_ = (ITKDatatypeOPort *)get_oport("Image1");
  if(!outport_) {
    error("Unable to initialize oport 'Image1'.");
    return;
  }

  typedef itk::ImageFileReader<itk::Image<float, 3> > FileReaderType;
  
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

void
 ImageReaderFloat3D::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight


