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

#include <SCICore/Persistent/Pstreams.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/VoidStarPort.h>
#include <SCICore/Datatypes/VoidStar.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class VoidStarWriter : public Module {
    VoidStarIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    VoidStarWriter(const clString& id);
    virtual ~VoidStarWriter();
    virtual void execute();
};

extern "C" Module* make_VoidStarWriter(const clString& id) {
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

VoidStarWriter::~VoidStarWriter()
{
}

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
    Pio(*stream, handle);
    delete stream;
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.7  2000/03/17 09:27:43  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.6  1999/10/07 02:07:13  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/25 03:48:16  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:18:02  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:20:16  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
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
// updates in Modules for Datatypes
//
// Revision 1.1  1999/04/25 03:25:36  dav
// adding these files in too... should have been there already... oh well, sigh
//
//
