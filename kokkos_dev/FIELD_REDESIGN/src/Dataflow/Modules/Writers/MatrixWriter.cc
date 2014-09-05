//static char *id="@(#) $Id$";

/*
 *  MatrixWriter.cc: Matrix Writer class
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
#include <PSECore/Datatypes/MatrixPort.h>
#include <SCICore/Datatypes/Matrix.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class MatrixWriter : public Module {
    MatrixIPort* inport;
    TCLstring filename;
    TCLstring filetype;
    TCLint split;
public:
    MatrixWriter(const clString& id);
    virtual ~MatrixWriter();
    virtual void execute();
};

extern "C" Module* make_MatrixWriter(const clString& id) {
  return new MatrixWriter(id);
}

MatrixWriter::MatrixWriter(const clString& id)
: Module("MatrixWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this), split("split", id, this)
{
    // Create the output data handle and port
    inport=scinew MatrixIPort(this, "Input Data", MatrixIPort::Atomic);
    add_iport(inport);
}

MatrixWriter::~MatrixWriter()
{
}

void MatrixWriter::execute()
{
    using SCICore::Containers::Pio;

    MatrixHandle handle;
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

    handle->set_raw( split.get() );

    Pio(*stream, handle);
    delete stream;
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.7.2.1  2000/10/31 02:29:37  dmw
// Merging PSECommon changes in HEAD into FIELD_REDESIGN branch
//
// Revision 1.8  2000/10/29 04:35:00  dmw
// BuildFEMatrix -- ground an arbitrary node
// SolveMatrix -- when preconditioning, be careful with 0's on diagonal
// MeshReader -- build the grid when reading
// SurfToGeom -- support node normals
// IsoSurface -- fixed tet mesh bug
// MatrixWriter -- support split file (header + raw data)
//
// LookupSplitSurface -- split a surface across a place and lookup values
// LookupSurface -- find surface nodes in a sfug and copy values
// Current -- compute the current of a potential field (- grad sigma phi)
// LocalMinMax -- look find local min max points in a scalar field
//
// Revision 1.7  2000/03/17 09:27:41  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.6  1999/10/07 02:07:12  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/25 03:48:15  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:18:01  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:20:15  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:56  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:20  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:32  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/27 22:58:04  dav
// updates in Modules for Datatypes
//
// Revision 1.1  1999/04/25 03:25:35  dav
// adding these files in too... should have been there already... oh well, sigh
//
//
