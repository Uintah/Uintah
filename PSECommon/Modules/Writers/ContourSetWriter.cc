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

#include <SCICore/Util/NotFinished.h>
#include <SCICore/Persistent/Pstreams.h>
#include <SCICore/CoreDatatypes/ContourSet.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <PSECore/Dataflow/Module.h>
#include <PSECore/CommonDatatypes/ContourSetPort.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class ContourSetWriter : public Module {
    ContourSetIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    ContourSetWriter(const clString& id);
    ContourSetWriter(const ContourSetWriter&, int deep=0);
    virtual ~ContourSetWriter();
    virtual Module* clone(int deep);
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

ContourSetWriter::ContourSetWriter(const ContourSetWriter& copy, int deep)
: Module(copy, deep), filename("filename", id, this),
  filetype("filetype", id, this)
{
    NOT_FINISHED("ContourSetWriter::ContourSetWriter");
}

ContourSetWriter::~ContourSetWriter()
{
}

Module* ContourSetWriter::clone(int deep)
{
    return scinew ContourSetWriter(*this, deep);
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
} // End namespace PSECommon

//
// $Log$
// Revision 1.2  1999/08/17 06:37:56  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:19  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:31  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/27 22:58:04  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1  1999/04/25 03:25:34  dav
// adding these files in too... should have been there already... oh well, sigh
//
//
