//static char *id="@(#) $Id$";

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

#include <DaveW/Datatypes/General/TensorField.h>
#include <DaveW/Datatypes/General/TensorFieldPort.h>
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
    using SCICore::Containers::Pio;

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

} // End namespace Modules
} // End namespace DaveW

//
// $Log$
// Revision 1.2  2000/03/17 09:26:06  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.1  1999/09/01 07:21:01  dmw
// new DaveW modules
//
//
