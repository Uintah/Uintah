
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

#include <Classlib/Pstreams.h>
#include <Dataflow/Module.h>
#include <Geom/Geom.h>
#include <Datatypes/GeometryPort.h>
#include <Malloc/Allocator.h>
#include <Datatypes/GeometryComm.h>
#include <TCL/TCLvar.h>

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

extern "C" {
  Module *make_GeometryReader(const clString& id)
  {
    return scinew GeometryReader(id);
  }
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

