//static char *id="@(#) $Id$";

/*
 *  VoidStarWriter.cc: VoidStar Writer class
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
#include <PSECore/Dataflow/Module.h>
#include <PSECore/CommonDatatypes/VoidStarPort.h>
#include <SCICore/CoreDatatypes/VoidStar.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class VoidStarWriter : public Module {
    VoidStarIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    VoidStarWriter(const clString& id);
    VoidStarWriter(const VoidStarWriter&, int deep=0);
    virtual ~VoidStarWriter();
    virtual Module* clone(int deep);
    virtual void execute();
};

Module* make_VoidStarWriter(const clString& id) {
  return new VoidStarWriter(id);
}

VoidStarWriter::VoidStarWriter(const clString& id)
: Module("VoidStarWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew VoidStarIPort(this, "Input Data", VoidStarIPort::Atomic);
    add_iport(inport);
}

VoidStarWriter::VoidStarWriter(const VoidStarWriter& copy, int deep)
: Module(copy, deep), filename("filename", id, this),
  filetype("filetype", id, this)
{
    NOT_FINISHED("VoidStarWriter::VoidStarWriter");
}

VoidStarWriter::~VoidStarWriter()
{
}

Module* VoidStarWriter::clone(int deep)
{
    return scinew VoidStarWriter(*this, deep);
}

#if 0
static void watcher(double pd, void* cbdata)
{
    VoidStarWriter* writer=(VoidStarWriter*)cbdata;
    writer->update_progress(pd);
}
#endif

void VoidStarWriter::execute()
{
    using SCICore::Containers::Pio;

    VoidStarHandle handle;
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
// Revision 1.2  1999/08/17 06:37:58  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:21  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:33  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/27 22:58:06  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1  1999/04/25 03:25:36  dav
// adding these files in too... should have been there already... oh well, sigh
//
//
