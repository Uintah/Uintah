
/*
 *  ColormapWriter.cc: Colormap Writer class
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
#include <Datatypes/ColormapPort.h>
#include <Datatypes/Colormap.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>

class ColormapWriter : public Module {
    ColormapIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    ColormapWriter(const clString& id);
    ColormapWriter(const ColormapWriter&, int deep=0);
    virtual ~ColormapWriter();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_ColormapWriter(const clString& id)
{
    return scinew ColormapWriter(id);
}
};

ColormapWriter::ColormapWriter(const clString& id)
: Module("ColormapWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew ColormapIPort(this, "Input Data", ColormapIPort::Atomic);
    add_iport(inport);
}

ColormapWriter::ColormapWriter(const ColormapWriter& copy, int deep)
: Module(copy, deep), filename("filename", id, this),
  filetype("filetype", id, this)
{
    NOT_FINISHED("ColormapWriter::ColormapWriter");
}

ColormapWriter::~ColormapWriter()
{
}

Module* ColormapWriter::clone(int deep)
{
    return scinew ColormapWriter(*this, deep);
}

static void watcher(double pd, void* cbdata)
{
    ColormapWriter* writer=(ColormapWriter*)cbdata;
    writer->update_progress(pd);
}

void ColormapWriter::execute()
{
    ColormapHandle handle;
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

template void Pio(Piostream&, ColormapHandle&);

#endif

#ifdef __sgi
#if _MIPS_SZPTR == 64
#include <Classlib/LockingHandle.cc>

static void _dummy_(Piostream& p1, ColormapHandle& p2)
{
    Pio(p1, p2);
}

#endif
#endif

