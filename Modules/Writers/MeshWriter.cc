
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

#include <Classlib/NotFinished.h>
#include <Classlib/Pstreams.h>
#include <Dataflow/Module.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/Mesh.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>

using sci::MeshHandle;

class MeshWriter : public Module {
    MeshIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    MeshWriter(const clString& id);
    MeshWriter(const MeshWriter&, int deep=0);
    virtual ~MeshWriter();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_MeshWriter(const clString& id)
{
    return scinew MeshWriter(id);
}
}

MeshWriter::MeshWriter(const clString& id)
: Module("MeshWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew MeshIPort(this, "Input Data", MeshIPort::Atomic);
    add_iport(inport);
}

MeshWriter::MeshWriter(const MeshWriter& copy, int deep)
: Module(copy, deep), filename("filename", id, this),
  filetype("filetype", id, this)
{
    NOT_FINISHED("MeshWriter::MeshWriter");
}

MeshWriter::~MeshWriter()
{
}

Module* MeshWriter::clone(int deep)
{
    return scinew MeshWriter(*this, deep);
}

#if 0
static void watcher(double pd, void* cbdata)
{
    MeshWriter* writer=(MeshWriter*)cbdata;
    writer->update_progress(pd);
}
#endif

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
    } else if (ft=="ASCII"){
	stream=scinew TextPiostream(fn, Piostream::Write);
    } else {	// GZIP!
	stream=scinew GzipPiostream(fn, Piostream::Write);
    }
    // Write the file
    //stream->watch_progress(watcher, (void*)this);
    Pio(*stream, handle);
    delete stream;
}
