
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

#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Network/Module.h>
#include <Core/Geom/GeomObj.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Ports/GeometryComm.h>
#include <Core/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace SCIRun {


class GeometryReader : public Module {

private:
  GeometryOPort *outport;
  TCLstring filename;
  
public:
  GeometryReader(const clString& id);
  virtual ~GeometryReader();
  virtual void execute();

};

extern "C" Module *make_GeometryReader(const clString& id) {
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

} // End namespace SCIRun

