
/*
 *  ReadField.cc: Reads Field datatype persistent objects
 *
 *  Written by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   December 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/Field.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLTask.h>
#include <Core/TclInterface/TCLvar.h>

namespace SCIRun {

class ReadField : public Module {

  // GROUP: private data
  //////////
  // 
  FieldOPort*       d_oport;
  TCLstring         d_filename;
  FieldHandle       d_hField;
  clString          d_oldFilename;

public:
  // GROUP: Constructor/Destructor
  //////////
  //
  ReadField(const clString& id);
  virtual ~ReadField();
  
  // GROUP: public member functions
  //////////
  //
  virtual void execute();
};

//////////
// Module maker function
extern "C" Module* make_ReadField(const clString& id) {
  return new ReadField(id);
}

//////////
// Constructor/Desctructor
ReadField::ReadField(const clString& id)
  : Module("ReadField", id, Source), d_filename("d_filename", id, this)
{
  d_oport=scinew FieldOPort(this, "Field Output", FieldIPort::Atomic);
  add_oport(d_oport);
}

ReadField::~ReadField()
{
}

//////////
// 
void ReadField::execute()
{
  clString fn(d_filename.get());
  
  if(!d_hField.get_rep() || fn != d_oldFilename){

    d_oldFilename=fn;
    Piostream* stream=auto_istream(fn);

    if(!stream){
      error(clString("Error reading file: ")+d_filename.get());
      return; // Can't open file...
    }
    
    // Read the file...
    Pio(*stream, d_hField);
    
    if(!d_hField.get_rep() || stream->error()){
      error("Error reading Field from file");
      delete stream;
      return;
    }
    delete stream;

  }

  d_oport->send(d_hField);
}

} // End namespace SCIRun

