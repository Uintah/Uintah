
/*
 *  TensorFieldWriter.cc: TensorField Writer class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Packages/DaveW/Core/Datatypes/General/TensorField.h>
#include <Packages/DaveW/Core/Datatypes/General/TensorFieldPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>

namespace DaveW {
using namespace DaveW::Datatypes;
using namespace SCIRun;

class TensorFieldWriter : public Module {
    TensorFieldIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    TensorFieldWriter(const clString& id);
    virtual ~TensorFieldWriter();
    virtual void execute();
};

extern "C" Module* make_TensorFieldWriter(const clString& id) {
  return new TensorFieldWriter(id);
}

TensorFieldWriter::TensorFieldWriter(const clString& id)
: Module("TensorFieldWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew TensorFieldIPort(this, "Input Data", TensorFieldIPort::Atomic);
    add_iport(inport);
}

TensorFieldWriter::~TensorFieldWriter()
{
}

#if 0
static void watcher(double pd, void* cbdata)
{
    TensorFieldWriter* writer=(TensorFieldWriter*)cbdata;
    writer->update_progress(pd);
}
#endif

void TensorFieldWriter::execute()
{

    TensorFieldHandle handle;
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
} // End namespace DaveW


