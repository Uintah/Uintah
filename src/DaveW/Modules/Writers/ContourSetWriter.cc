//static char *id="@(#) $Id$";

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

#include <DaveW/Datatypes/General/ContourSet.h>
#include <DaveW/Datatypes/General/ContourSetPort.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Persistent/Pstreams.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace DaveW {
namespace Modules {

using namespace DaveW::Datatypes;
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class ContourSetWriter : public Module {
    ContourSetIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    ContourSetWriter(const clString& id);
    virtual ~ContourSetWriter();
    virtual void execute();
};

Module* make_ContourSetWriter(const clString& id) {
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
    using SCICore::Containers::Pio;

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

} // End namespace Modules
} // End namespace DaveW

//
// $Log$
// Revision 1.1  1999/09/01 07:21:00  dmw
// new DaveW modules
//
//
