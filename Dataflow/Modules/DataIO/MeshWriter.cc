
/*
 *  MeshWriter.cc: Mesh Writer class
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
#include <Dataflow/Ports/MeshPort.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>

namespace SCIRun {


class MeshWriter : public Module {
    MeshIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    MeshWriter(const clString& id);
    virtual ~MeshWriter();
    virtual void execute();
};

extern "C" Module* make_MeshWriter(const clString& id) {
  return new MeshWriter(id);
}

MeshWriter::MeshWriter(const clString& id)
: Module("MeshWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew MeshIPort(this, "Input Data", MeshIPort::Atomic);
    add_iport(inport);
}

MeshWriter::~MeshWriter()
{
}

void MeshWriter::execute()
{

    MeshHandle handle;
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

