
/*
 *  NrrdReader.cc: Read in a Nrrd
 *
 *  Written by:
 *   David Weinstein
 *   School of Computing
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>
#include <Nrrd/Dataflow/Ports/NrrdPort.h>

namespace SCINrrd {

using namespace SCIRun;

class NrrdReader : public Module {
    NrrdOPort* outport;
    TCLstring filename;
    NrrdDataHandle handle;
    clString old_filename;
public:
    NrrdReader(const clString& id);
    virtual ~NrrdReader();
    virtual void execute();
};

extern "C" Module* make_NrrdReader(const clString& id) {
  return new NrrdReader(id);
}

NrrdReader::NrrdReader(const clString& id)
: Module("NrrdReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew NrrdOPort(this, "Output Data", NrrdIPort::Atomic);
    add_oport(outport);
}

NrrdReader::~NrrdReader()
{
}

void NrrdReader::execute()
{

    clString fn(filename.get());
    if(!handle.get_rep() || fn != old_filename){
	old_filename=fn;
	Piostream* stream=auto_istream(fn);
	if(!stream){
	    error(clString("Error reading file: ")+filename.get());
	    return; // Can't open file...
	}
	// Read the file...
	Pio(*stream, handle);
	if(!handle.get_rep() || stream->error()){
	    error("Error reading Nrrd from file");
	    delete stream;
	    return;
	}
	delete stream;
    }
    outport->send(handle);
}

} // End namespace SCINrrd
