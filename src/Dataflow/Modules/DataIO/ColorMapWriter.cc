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
 *  ColorMapWriter.cc: Save persistent representation of a colormap to a file
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColorMapPort.h>

namespace SCIRun {

class ColorMapWriter : public Module {
  ColorMapIPort* inport_;
  GuiString filename_;
  GuiString filetype_;
public:
  ColorMapWriter(const string& id);
  virtual ~ColorMapWriter();
  virtual void execute();
};

extern "C" Module* make_ColorMapWriter(const string& id) {
  return new ColorMapWriter(id);
}

ColorMapWriter::ColorMapWriter(const string& id)
  : Module("ColorMapWriter", id, Source, "DataIO", "SCIRun"),
    filename_("filename", id, this),
    filetype_("filetype", id, this)
{
  // Create the output port
  inport_=scinew ColorMapIPort(this, "Persistent Data", ColorMapIPort::Atomic);
  add_iport(inport_);
}

ColorMapWriter::~ColorMapWriter()
{
}

void ColorMapWriter::execute()
{
  // Read data from the input port
  ColorMapHandle handle;
  if(!inport_->get(handle))
    return;

  // If no name is provided, return
  const string fn(filename_.get());
  if (fn == "") {
    warning("No filename.");
    return;
  }
   
  // Open up the output stream
  Piostream* stream;
  string ft(filetype_.get());
  if(ft=="Binary") {
    stream=scinew BinaryPiostream(fn, Piostream::Write);
  } else { // "ASCII"
    stream=scinew TextPiostream(fn, Piostream::Write);
  }

  // Write the file
  Pio(*stream, handle);
  delete stream;
}

} // End namespace SCIRun
