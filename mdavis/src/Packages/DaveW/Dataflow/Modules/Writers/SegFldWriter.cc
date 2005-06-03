
/*
 *  SegFldWriter.cc: SegFld Writer class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Packages/DaveW/Core/Datatypes/General/SegFld.h>
#include <Packages/DaveW/Core/Datatypes/General/SegFldPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>

namespace DaveW {
using namespace DaveW;
using namespace SCIRun;

class SegFldWriter : public Module {
    SegFldIPort* inport;
    GuiString filename;
    GuiString filetype;
public:
    SegFldWriter(const clString& id);
    virtual ~SegFldWriter();
    virtual void execute();
};

extern "C" Module* make_SegFldWriter(const clString& id) {
  return new SegFldWriter(id);
}

SegFldWriter::SegFldWriter(const clString& id)
: Module("SegFldWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew SegFldIPort(this, "Input Data", SegFldIPort::Atomic);
    add_iport(inport);
}

SegFldWriter::~SegFldWriter()
{
}

#if 0
static void watcher(double pd, void* cbdata)
{
    SegFldWriter* writer=(SegFldWriter*)cbdata;
    writer->update_progress(pd);
}
#endif

void SegFldWriter::execute()
{

    SegFldHandle handle;
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


