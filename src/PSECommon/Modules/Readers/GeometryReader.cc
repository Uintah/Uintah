//static char *id="@(#) $Id$";

/*
 *  GeometryReader.cc: Geometry Reader class
 *    Reads in a GeomObj from a file.
 *
 *  Written by:
 *   Philip Sutton
 *   Department of Computer Science
 *   University of Utah
 *   October 1998
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <SCICore/Persistent/Pstreams.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Geom/GeomObj.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <SCICore/Malloc/Allocator.h>
#include <PSECore/Datatypes/GeometryComm.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::PersistentSpace;

class GeometryReader : public Module {

private:
  GeometryOPort *outport;
  TCLstring filename;
  
public:
  GeometryReader(const clString& id);
  virtual ~GeometryReader();
  virtual void execute();

};

Module *make_GeometryReader(const clString& id) {
  return new GeometryReader(id);
}

GeometryReader::GeometryReader(const clString& id)
  : Module("GeometryReader", id, Source), filename("filename", id, this)
{
  outport = scinew GeometryOPort(this, "Output Data", GeometryIPort::Atomic);
  add_oport(outport);
}

GeometryReader::~GeometryReader()
{
}

void GeometryReader::execute() {
  clString fn( filename.get() );
  Piostream *stream = auto_istream(fn);
  GeomObj *obj;
  if( !stream ) {
    cerr << "Error reading file: " << filename.get() << endl;
    return; // Can't open file...
  }

  // Read the file
  Pio(*stream, obj );
  if( stream->error() ) {
    cerr << "Error reading Geometry from file" << endl;
    delete stream;
    return;
  }

  delete stream;
  outport->addObj( obj, "Geometry" );
  outport->flush();
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.4  1999/08/25 03:47:54  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.3  1999/08/18 20:19:48  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:34  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:48  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//
