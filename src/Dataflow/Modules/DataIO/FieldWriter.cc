/*
 *  FieldWriter.cc: Save persistent representation of a field to a file
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
#include <Dataflow/Ports/FieldPort.h>

namespace SCIRun {

class FieldWriter : public Module {
  FieldIPort* inport_;
  GuiString filename_;
  GuiString filetype_;
public:
  FieldWriter(const clString& id);
  virtual ~FieldWriter();
  virtual void execute();
};

extern "C" Module* make_FieldWriter(const clString& id) {
  return new FieldWriter(id);
}

FieldWriter::FieldWriter(const clString& id)
  : Module("FieldWriter", id, Source), filename_("filename", id, this),
    filetype_("filetype", id, this)
{
  // Create the output port
  inport_=scinew FieldIPort(this, "Persistent Data", FieldIPort::Atomic);
  add_iport(inport_);
}

FieldWriter::~FieldWriter()
{
}

void FieldWriter::execute()
{
  // Read data from the input port
  FieldHandle handle;
  if(!inport_->get(handle))
    return;

  // If no name is provided, return
  clString fn(filename_.get());
  if(fn == "") {
    error("Warning: no filename in FieldWriter");
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
