/*
 *  FieldReader.cc: Read a persistent field from a file
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

class FieldReader : public Module {
    FieldOPort* outport_;
    GuiString filename_;
    FieldHandle handle_;
    clString old_filename_;
public:
    FieldReader(const clString& id);
    virtual ~FieldReader();
    virtual void execute();
};

extern "C" Module* make_FieldReader(const clString& id) {
  return new FieldReader(id);
}

FieldReader::FieldReader(const clString& id)
: Module("FieldReader", id, Source), filename_("filename", id, this)
{
    // Create the output port
    outport_=scinew FieldOPort(this, "Output Data", FieldIPort::Atomic);
    add_oport(outport_);
}

FieldReader::~FieldReader()
{
}

void FieldReader::execute()
{
    clString fn(filename_.get());

    // If we haven't read yet, or if it's a new filename, then read
    if(!handle_.get_rep() || fn != old_filename_){
	old_filename_=fn;
	Piostream* stream=auto_istream(fn);
	if(!stream){
	    error(clString("Error reading file: ")+filename_.get());
	    return;
	}

	// Read the file
	Pio(*stream, handle_);
	if(!handle_.get_rep() || stream->error()){
	    error(clString("Error reading Field from file: ")+
		  filename_.get());
	    delete stream;
	    return;
	}
	delete stream;
    }

    // Send the data downstream
    outport_->send(handle_);
}

} // End namespace SCIRun
