
/*
 *  ContourSetWriter.cc: ContourSet Writer class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Packages/DaveW/Core/Datatypes/General/ContourSet.h>
#include <Packages/DaveW/Core/Datatypes/General/ContourSetPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>

namespace DaveW {
using namespace DaveW;
using namespace SCIRun;

class ContourSetWriter : public Module {
    ContourSetIPort* inport;
    GuiString filename;
    GuiString filetype;
public:
    ContourSetWriter(const clString& id);
    virtual ~ContourSetWriter();
    virtual void execute();
};

extern "C" Module* make_ContourSetWriter(const clString& id) {
  return new ContourSetWriter(id);
}

ContourSetWriter::ContourSetWriter(const clString& id)
: Module("ContourSetWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew ContourSetIPort(this, "Input Data", ContourSetIPort::Atomic);
    add_iport(inport);
}

ContourSetWriter::~ContourSetWriter()
{
}

#if 0
static void watcher(double pd, void* cbdata)
{
    ContourSetWriter* writer=(ContourSetWriter*)cbdata;
    writer->update_progress(pd);
}
#endif

void ContourSetWriter::execute()
{

    ContourSetHandle handle;
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


