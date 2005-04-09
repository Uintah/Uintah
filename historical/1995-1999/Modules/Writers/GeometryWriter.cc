
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

#include <Classlib/Pstreams.h>
#include <Dataflow/Module.h>
#include <Geom/Geom.h>
#include <Datatypes/GeometryPort.h>
#include <Malloc/Allocator.h>
#include <Datatypes/GeometryComm.h>
#include <TCL/TCLvar.h>

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
  GeometryWriter(const GeometryWriter&, int deep=0);
  virtual ~GeometryWriter();
  virtual Module* clone(int deep);
  virtual void execute();

};

extern "C" {
  Module *make_GeometryWriter(const clString& id)
  {
    return scinew GeometryWriter(id);
  }
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

GeometryWriter::GeometryWriter(const GeometryWriter& copy, int deep)
  : Module( copy, deep ), filename("filename", id, this),
    filetype("filetype", id, this)
{
}

GeometryWriter::~GeometryWriter()
{
}

Module *GeometryWriter::clone(int deep) {
  return scinew GeometryWriter(*this,deep);
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
      Pio(*stream, gmsg->obj);
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

