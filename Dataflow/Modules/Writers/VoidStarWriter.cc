
/*
 *  VoidStarWriter.cc: VoidStar Writer class
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
#include <Dataflow/Ports/VoidStarPort.h>
#include <Core/Datatypes/VoidStar.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>

namespace SCIRun {


class VoidStarWriter : public Module {
    VoidStarIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    VoidStarWriter(const clString& id);
    virtual ~VoidStarWriter();
    virtual void execute();
};

extern "C" Module* make_VoidStarWriter(const clString& id) {
  return new VoidStarWriter(id);
}

VoidStarWriter::VoidStarWriter(const clString& id)
: Module("VoidStarWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew VoidStarIPort(this, "Input Data", VoidStarIPort::Atomic);
    add_iport(inport);
}

VoidStarWriter::~VoidStarWriter()
{
}

void VoidStarWriter::execute()
{

    VoidStarHandle handle;
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

