
/*
 *  SigmaSetWriter.cc: SigmaSet Writer class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Packages/DaveW/Core/Datatypes/General/SigmaSet.h>
#include <Packages/DaveW/Core/Datatypes/General/SigmaSetPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/GuiInterface/GuiVar.h>

namespace DaveW {
using namespace DaveW;
using namespace SCIRun;

class SigmaSetWriter : public Module {
    SigmaSetIPort* inport;
    GuiString filename;
    GuiString filetype;
public:
    SigmaSetWriter(const clString& get_id());
    virtual ~SigmaSetWriter();
    virtual void execute();
};

extern "C" Module* make_SigmaSetWriter(const clString& get_id()) {
  return new SigmaSetWriter(get_id());
}

SigmaSetWriter::SigmaSetWriter(const clString& get_id())
: Module("SigmaSetWriter", get_id(), Source), filename("filename", get_id(), this),
  filetype("filetype", get_id(), this)
{
    // Create the output data handle and port
    inport=scinew SigmaSetIPort(this, "Input Data", SigmaSetIPort::Atomic);
    add_iport(inport);
}

SigmaSetWriter::~SigmaSetWriter()
{
}

#if 0
static void watcher(double pd, void* cbdata)
{
    SigmaSetWriter* writer=(SigmaSetWriter*)cbdata;
    writer->update_progress(pd);
}
#endif

void SigmaSetWriter::execute()
{

    SigmaSetHandle handle;
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


