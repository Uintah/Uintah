/*
 *  PathReader.cc: Read a persistent camera path from a file
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
#include <Dataflow/Ports/PathPort.h>

namespace SCIRun {

class PathReader : public Module {
    PathOPort* outport_;
    GuiString filename_;
    PathHandle handle_;
    clString old_filename_;
public:
    PathReader(const clString& id);
    virtual ~PathReader();
    virtual void execute();
};

extern "C" Module* make_PathReader(const clString& id) {
  return new PathReader(id);
}

PathReader::PathReader(const clString& id)
: Module("PathReader", id, Source), filename_("filename", id, this)
{
    // Create the output port
    outport_=scinew PathOPort(this, "Output Data", PathIPort::Atomic);
    add_oport(outport_);
}

PathReader::~PathReader()
{
}

void PathReader::execute()
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
	    error(clString("Error reading Path from file: ")+
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
