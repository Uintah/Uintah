
/*
 *  VectorFieldWriter.cc: VectorField Writer class
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
#include <Dataflow/Ports/VectorFieldPort.h>
#include <Core/Datatypes/VectorField.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>

namespace SCIRun {


class VectorFieldWriter : public Module {
    VectorFieldIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    VectorFieldWriter(const clString& id);
    virtual ~VectorFieldWriter();
    virtual void execute();
};

extern "C" Module* make_VectorFieldWriter(const clString& id) {
  return new VectorFieldWriter(id);
}

VectorFieldWriter::VectorFieldWriter(const clString& id)
: Module("VectorFieldWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew VectorFieldIPort(this, "Input Data", VectorFieldIPort::Atomic);
    add_iport(inport);
}

VectorFieldWriter::~VectorFieldWriter()
{
}

void VectorFieldWriter::execute()
{

    VectorFieldHandle handle;
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

