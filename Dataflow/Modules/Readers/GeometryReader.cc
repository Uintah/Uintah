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
#include <PSECore/CommonDatatypes/GeometryPort.h>
#include <SCICore/Malloc/Allocator.h>
#include <PSECore/CommonDatatypes/GeometryComm.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::PersistentSpace;

class GeometryReader : public Module {

private:
  GeometryOPort *outport;
  TCLstring filename;
  
public:
  GeometryReader(const clString& id);
  GeometryReader(const GeometryReader&, int deep=0);
  virtual ~GeometryReader();
  virtual Module* clone(int deep);
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

GeometryReader::GeometryReader(const GeometryReader& copy, int deep)
  : Module( copy, deep ), filename("filename", id, this)
{
}

GeometryReader::~GeometryReader()
{
}

Module *GeometryReader::clone(int deep) {
  return scinew GeometryReader(*this,deep);
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
