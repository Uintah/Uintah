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
  ColorMapWriter(const clString& id);
  virtual ~ColorMapWriter();
  virtual void execute();
};

extern "C" Module* make_ColorMapWriter(const clString& id) {
  return new ColorMapWriter(id);
}

ColorMapWriter::ColorMapWriter(const clString& id)
  : Module("ColorMapWriter", id, Source), filename_("filename", id, this),
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
  clString fn(filename_.get());
  if(fn == "") {
    error("Warning: no filename in ColorMapWriter");
    return;
  }
   
  // Open up the output stream
  Piostream* stream;
  clString ft(filetype_.get());
  if(ft=="Binary"){
    stream=scinew BinaryPiostream(fn, Piostream::Write);
  } else { // "ASCII"
    stream=scinew TextPiostream(fn, Piostream::Write);
  }

  // Write the file
  Pio(*stream, handle);
  delete stream;
}

} // End namespace SCIRun
