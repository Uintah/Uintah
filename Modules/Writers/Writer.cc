
/*
 *  TYPEWriter.cc: TYPE Writer class
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
#include <Dataflow/ModuleList.h>
#include <Datatypes/TYPEPort.h>
#include <Datatypes/TYPE.h>
#include <TCL/TCLvar.h>

class TYPEWriter : public Module {
    TYPEIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    TYPEWriter(const clString& id);
    TYPEWriter(const TYPEWriter&, int deep=0);
    virtual ~TYPEWriter();
    virtual Module* clone(int deep);
    virtual void execute();
};

static Module* make_TYPEWriter(const clString& id)
{
    return new TYPEWriter(id);
}

#include "TYPERegister.h"

TYPEWriter::TYPEWriter(const clString& id)
: Module("TYPEWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=new TYPEIPort(this, "Input Data", TYPEIPort::Atomic);
    add_iport(inport);
}

TYPEWriter::TYPEWriter(const TYPEWriter& copy, int deep)
: Module(copy, deep), filename("filename", id, this),
  filetype("filetype", id, this)
{
    NOT_FINISHED("TYPEWriter::TYPEWriter");
}

TYPEWriter::~TYPEWriter()
{
}

Module* TYPEWriter::clone(int deep)
{
    return new TYPEWriter(*this, deep);
}

static void watcher(double pd, void* cbdata)
{
    TYPEWriter* writer=(TYPEWriter*)cbdata;
    writer->update_progress(pd);
}

void TYPEWriter::execute()
{
    TYPEHandle handle;
    if(!inport->get(handle))
	return;
    clString fn(filename.get());
    if(fn == "")
	return;
    Piostream* stream;
    clString ft(filetype.get());
    if(ft=="Binary"){
	stream=new BinaryPiostream(fn, Piostream::Write);
    } else {
	stream=new TextPiostream(fn, Piostream::Write);
    }
    // Write the file
    //stream->watch_progress(watcher, (void*)this);
    Pio(*stream, handle);
    delete stream;
}
