
/*
 *  FieldReader.cc: Reads Field datatype persistent objects
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

class FieldReader : public Module {

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
  FieldReader(const clString& id);
  virtual ~FieldReader();
  
  // GROUP: public member functions
  //////////
  //
  virtual void execute();
};

//////////
// Module maker function
extern "C" Module* make_FieldReader(const clString& id) {
  return new FieldReader(id);
}

//////////
// Constructor/Desctructor
FieldReader::FieldReader(const clString& id)
  : Module("FieldReader", id, Source), d_filename("d_filename", id, this)
{
  d_oport=scinew FieldOPort(this, "Field Output", FieldIPort::Atomic);
  add_oport(d_oport);
}

FieldReader::~FieldReader()
{
}

//////////
// 
void FieldReader::execute()
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

