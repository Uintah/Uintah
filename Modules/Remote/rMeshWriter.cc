
/*
 *  rMeshWriter.cc: remote Mesh Writer class
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

class rMeshWriter : public Module {
    MeshIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    rMeshWriter(const clString& id);
    rMeshWriter(const rMeshWriter&, int deep=0);
    virtual ~rMeshWriter();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_rMeshWriter(const clString& id)
{
    return scinew rMeshWriter(id);
}
};

rMeshWriter::rMeshWriter(const clString& id)
: Module("rMeshWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew MeshIPort(this, "Input Data", MeshIPort::Atomic);
    add_iport(inport);
}

rMeshWriter::rMeshWriter(const rMeshWriter& copy, int deep)
: Module(copy, deep), filename("filename", id, this),
  filetype("filetype", id, this)
{
    NOT_FINISHED("rMeshWriter::rMeshWriter");
}

rMeshWriter::~rMeshWriter()
{
}

Module* rMeshWriter::clone(int deep)
{
    return scinew rMeshWriter(*this, deep);
}

static void watcher(double pd, void* cbdata)
{
    rMeshWriter* writer=(rMeshWriter*)cbdata;
    writer->update_progress(pd);
}

void rMeshWriter::execute()
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
    //stream->watch_progress(watcher, (void*)this);
    Pio(*stream, handle);
    delete stream;
}

#ifdef __GNUG__

#include <Classlib/LockingHandle.cc>

template void Pio(Piostream&, MeshHandle&);

#endif

#ifdef __sgi
#if _MIPS_SZPTR == 64
#include <Classlib/LockingHandle.cc>

static void _dummy_(Piostream& p1, MeshHandle& p2)
{
    Pio(p1, p2);
}

#endif
#endif

