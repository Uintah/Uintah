/*
 *  NrrdWriter.cc: Nrrd writer
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Network/Module.h>
#include <Nrrd/Dataflow/Ports/NrrdPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>

namespace SCINrrd {

using namespace SCIRun;

class NrrdWriter : public Module {
    NrrdIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    NrrdWriter(const clString& id);
    virtual ~NrrdWriter();
    virtual void execute();
};

extern "C" Module* make_NrrdWriter(const clString& id) {
  return new NrrdWriter(id);
}

NrrdWriter::NrrdWriter(const clString& id)
: Module("NrrdWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew NrrdIPort(this, "Input Data", NrrdIPort::Atomic);
    add_iport(inport);
}

NrrdWriter::~NrrdWriter()
{
}

void NrrdWriter::execute()
{

    NrrdDataHandle handle;
    if(!inport->get(handle))
	return;
    clString fn(filename.get());
    if(fn == "")
	return;
    Piostream* stream;
    clString ft(filetype.get());
    if(ft=="Binary"){
	stream=scinew BinaryPiostream(fn, Piostream::Write);
    } else {
	stream=scinew TextPiostream(fn, Piostream::Write);
    }
    // Write the file
    Pio(*stream, handle);
    delete stream;
}

} // End namespace SCINrrd

