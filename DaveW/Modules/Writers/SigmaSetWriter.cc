//static char *id="@(#) $Id$";

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

#include <DaveW/Datatypes/General/SigmaSet.h>
#include <DaveW/Datatypes/General/SigmaSetPort.h>
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

class SigmaSetWriter : public Module {
    SigmaSetIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    SigmaSetWriter(const clString& id);
    virtual ~SigmaSetWriter();
    virtual void execute();
};

Module* make_SigmaSetWriter(const clString& id) {
  return new SigmaSetWriter(id);
}

SigmaSetWriter::SigmaSetWriter(const clString& id)
: Module("SigmaSetWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
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
    using SCICore::Containers::Pio;

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

} // End namespace Modules
} // End namespace DaveW

//
// $Log$
// Revision 1.1  1999/09/01 07:21:01  dmw
// new DaveW modules
//
//
