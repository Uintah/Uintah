
/*
 *  SurfaceWriter.cc: Surface Writer class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Core/Datatypes/Surface.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>

namespace SCIRun {


class SurfaceWriter : public Module {
    SurfaceIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    SurfaceWriter(const clString& id);
    virtual ~SurfaceWriter();
    virtual void execute();
};

extern "C" Module* make_SurfaceWriter(const clString& id) {
  return new SurfaceWriter(id);
}

SurfaceWriter::SurfaceWriter(const clString& id)
: Module("SurfaceWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew SurfaceIPort(this, "Input Data", SurfaceIPort::Atomic);
    add_iport(inport);
}

SurfaceWriter::~SurfaceWriter()
{
}

void SurfaceWriter::execute()
{

    SurfaceHandle handle;
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

} // End namespace SCIRun

