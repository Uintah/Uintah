//  Extractor.cc - Extracts a field, attribute, or geometry from a domain 
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   March 2000
//  Copyright (C) 2000 SCI Group
//  Module Extractor
//  Description: 
//  Input ports: 
//  Output ports:

#include <stdio.h>
#include <Core/TclInterface/TCLvar.h>

#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/DomainPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/AttribPort.h>
#include <Dataflow/Ports/GeomPort.h>


namespace SCIRun {

    
class Extractor : public Module {
private:
  DomainIPort* iport;
  AttribOPort* aoport;
  GeomOPort* goport;
  FieldOPort* foport;
public:
  Extractor(const clString& id);
  virtual ~Extractor();
  virtual void execute();
};

extern "C" Module* make_Extractor(const clString& id){
  return new Extractor(id);
}

Extractor::Extractor(const clString& id)
  : Module("Extractor", id, Filter)
  {
    // Create the input port
    iport = new DomainIPort(this, "Domain", DomainIPort::Atomic);
    add_iport(iport);
    // Create the output port
    foport = new FieldOPort(this, "Field", FieldIPort::Atomic);
    add_oport(foport);
    goport = new GeomOPort(this, "Geom", GeomIPort::Atomic);
    add_oport(goport);
    aoport = new AttribOPort(this, "Attrib", AttribIPort::Atomic);
    add_oport(aoport);
    
  }

Extractor::~Extractor()
  {
  }

void Extractor::execute()
  {
    
  }

} // End namespace SCIRun

