//static char *id="@(#) $Id$";

/*
 *  ScalarFieldWriter.cc: ScalarField Writer class
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
#include <PSECore/CommonDatatypes/ScalarFieldPort.h>
#include <SCICore/CoreDatatypes/ScalarField.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class ScalarFieldWriter : public Module {
    ScalarFieldIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    ScalarFieldWriter(const clString& id);
    ScalarFieldWriter(const ScalarFieldWriter&, int deep=0);
    virtual ~ScalarFieldWriter();
    virtual Module* clone(int deep);
    virtual void execute();
};

Module* make_ScalarFieldWriter(const clString& id) {
  return new ScalarFieldWriter(id);
}

ScalarFieldWriter::ScalarFieldWriter(const clString& id)
: Module("ScalarFieldWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew ScalarFieldIPort(this, "Input Data", ScalarFieldIPort::Atomic);
    add_iport(inport);
}

ScalarFieldWriter::ScalarFieldWriter(const ScalarFieldWriter& copy, int deep)
: Module(copy, deep), filename("filename", id, this),
  filetype("filetype", id, this)
{
    NOT_FINISHED("ScalarFieldWriter::ScalarFieldWriter");
}

ScalarFieldWriter::~ScalarFieldWriter()
{
}

Module* ScalarFieldWriter::clone(int deep)
{
    return scinew ScalarFieldWriter(*this, deep);
}

#if 0
static void watcher(double pd, void* cbdata)
{
    ScalarFieldWriter* writer=(ScalarFieldWriter*)cbdata;
    writer->update_progress(pd);
}
#endif

void ScalarFieldWriter::execute()
{
    using SCICore::Containers::Pio;

    ScalarFieldHandle handle;
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
// Revision 1.2  1999/08/17 06:37:57  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:20  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:32  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/27 22:58:05  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1  1999/04/25 03:25:35  dav
// adding these files in too... should have been there already... oh well, sigh
//
//
