
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

#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Dataflow/Module.h>
#include <Dataflow/Datatypes/TYPEPort.h>
#include <Dataflow/Datatypes/TYPE.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>

namespace SCIRun {


class TYPEWriter : public Module {
    TYPEIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    TYPEWriter(const clString& id);
    virtual ~TYPEWriter();
    virtual void execute();
};

extern "C" Module* make_TYPEWriter(const clString& id) {
  return new TYPEWriter(id);
}

TYPEWriter::TYPEWriter(const clString& id)
: Module("TYPEWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew TYPEIPort(this, "Input Data", TYPEIPort::Atomic);
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
	stream=scinew BinaryPiostream(fn, Piostream::Write);
    } else {
	stream=scinew TextPiostream(fn, Piostream::Write);
    }
    // Write the file
    Pio(*stream, handle);
    delete stream;
}

} // End namespace SCIRun

