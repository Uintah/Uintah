//  Extractor.cc - Extracts a field, attribute, or geometry from a domain 
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   March 2000
//
//  Copyright (C) 2000 SCI Group
//
//  Module Extractor
//
//  Description: 
//  Input ports: 
//  Output ports:

#include <stdio.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <SCICore/Malloc/Allocator.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/DomainPort.h>
#include <PSECore/Datatypes/FieldPort.h>
#include <PSECore/Datatypes/AttribPort.h>
#include <PSECore/Datatypes/GeomPort.h>


namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
    
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

} // End namespace Modules
} // End namespace PSECommon

