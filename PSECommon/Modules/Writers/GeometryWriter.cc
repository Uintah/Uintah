//static char *id="@(#) $Id$";

/*
 *  GeometryWriter.cc: Geometry Writer class
 *    Writes a GeomObj to a file
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
#include <iostream>
using std::cerr;
using std::endl;

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class GeometryWriter : public Module {

private:
  GeometryIPort *inport;
  TCLstring filename;
  TCLstring filetype;
  int portid, busy;
  int done;
  
  virtual void do_execute();
  void process_event();

public:
  GeometryWriter(const clString& id);
  virtual ~GeometryWriter();
  virtual void execute();

};

Module *make_GeometryWriter(const clString& id) {
  return new GeometryWriter(id);
}

GeometryWriter::GeometryWriter(const clString& id)
  : Module("GeometryWriter", id, Source), filename("filename", id, this),
    filetype("filetype", id, this)
{
  inport = scinew GeometryIPort(this, "Input Data", GeometryIPort::Atomic);
  add_iport(inport);
  done = 0;
  busy = 0;
  have_own_dispatch = 1; 
}

GeometryWriter::~GeometryWriter()
{
}

void GeometryWriter::do_execute() {
  while( !done ) {
    process_event();
  }
  update_state( Completed );
}

void GeometryWriter::process_event() {
  MessageBase* msg=mailbox.receive();
  GeometryComm* gmsg=(GeometryComm*)msg;

  switch(gmsg->type){
  case MessageTypes::ExecuteModule:
    // We ignore these messages...
    break;
  case MessageTypes::GeometryAddObj:
    {
      filename.reset();
      filetype.reset();
      update_state( JustStarted );
      
      clString fn(filename.get());
      if( fn == "" ) {
	cerr << "GeometryWriter Error: filename empty" << endl;
	done = 1;
	return;
      }

      Piostream *stream;
      clString ft(filetype.get());
      if( ft=="Binary" ) {
	stream = scinew BinaryPiostream(fn, Piostream::Write);
      } else {
	stream=scinew TextPiostream(fn, Piostream::Write);
      }

      // Write the file
      SCICore::GeomSpace::Pio(*stream, gmsg->obj);
      delete stream;
      done = 1;
    }
    break;
  case MessageTypes::GeometryInit:
    gmsg->reply->send(GeomReply(portid++, &busy));
    break;	
  case MessageTypes::GeometryDelObj:
  case MessageTypes::GeometryDelAll:
  case MessageTypes::GeometryFlush:
  case MessageTypes::GeometryFlushViews:
  case MessageTypes::GeometryGetNRoe:
  case MessageTypes::GeometryGetData:
    break;
  default:
    cerr << "GeometryWriter: Illegal Message type: " << gmsg->type << endl;
    break;
  }
}

void GeometryWriter::execute() {
  // never gets called
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.5  1999/10/07 02:07:12  sparker
// use standard iostreams and complex type
//
// Revision 1.4  1999/08/25 03:48:14  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
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
// Revision 1.1  1999/07/27 16:58:19  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:33  dav
// Import sources
//
//
