// Register.cc - Registers a field, attribute, or geometry with a domain.
 //
 //  Written by:
 //   Eric Kuehne
 //   Department of Computer Science
 //   University of Utah
 //   April 2000
 //
 //  Copyright (C) 2000 SCI Group
 //
 //  Module Register
 //
 //  Description: This module takes a field, attribute, or geometry as
 //  input and sends a FieldWrapper to output.  Duties include:
 //     - Create a fill in the FieldWrapper object appropriatly
 //     according to user input
 //  Input ports: 
 //  Output ports:
 //
// TODO: Add support for geom and attrib inports
#include <stdio.h>

#include <SCICore/TclInterface/TCLvar.h>

#include <SCICore/Malloc/Allocator.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/FieldWrapperPort.h>
#include <PSECore/Datatypes/FieldPort.h>
#include <SCICore/Datatypes/Domain.h>
#include <SCICore/Datatypes/FieldWrapper.h>
#include <PSECommon/Modules/Domain/DomainManager.h>
#include <SCICore/Util/DebugStream.h>
#include <SCICore/Containers/String.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Util;

class Register : public Module {
private:
  FieldIPort* ifield;
  FieldWrapperOPort* oport;

  TCLstring tcl_attrib;
  TCLstring tcl_geom;

  FieldWrapperHandle fwh;
  FieldWrapper *fw;
  FieldHandle sfh;  

  DebugStream dbg;
public:
  Register(const clString& id);
  virtual ~Register();
  virtual void execute();
  void connection(ConnectionMode, int, int);
  
  DomainHandle domain;
};

extern "C" Module* make_Register(const clString& id){
  return new Register(id);
}

Register::Register(const clString& id)
  : Module("Register", id, Filter),dbg("Register", true),
  tcl_geom("geom", id, this), tcl_attrib("attrib", id, this)
  {
    ifield = new FieldIPort(this, "GenField", FieldIPort::Atomic);
    add_iport(ifield);
    // the which_port attribute is used by the connection funciton below
    // have to set which_port *AFTER* add_Xport is called...
    oport = new FieldWrapperOPort(this, "FieldWrapper", FieldWrapperIPort::Atomic);
    add_oport(oport);
  }

Register::~Register()
  {
  }

void Register::execute(){
  // Check to see if we have anything on in-ports
  dbg << "Register::execute is running..." << endl;



  sfh = 0;
  if(ifield->get(sfh)){
    // get the user input
    dbg << "ifield->get(sfh)" << endl;
    string attrib = tcl_attrib.get()();
    string geom = tcl_geom.get()();
    if(sfh->status == NEW){
      dbg << "sfh->status == NEW" << endl;
      dbg << "attrib.length: " << attrib.length() << " geom.length: " << geom.length() << endl;
      if((attrib.length() == 0) || (geom.length() == 0)){
      // TODO: fix this...the error message doesn't appear for some reason
	error("You must specify a name for both the attribute and the geometry.");
	return;
      }
      sfh->get_geom()->setName(geom);
      sfh->get_attrib()->setName(attrib);
      fwh = fw = 0;
      fwh = fw = new FieldWrapper(sfh, NEW);
      oport->send(fw);
    }
    else if(sfh->status == OLD){
    }
    else if(sfh->status == SHARED){
      
    }
  }
  else{ // There was nothing on the in-ports
    return;
  }

  
  
}

void Register::connection(ConnectionMode mode, int, int is_oport){
  if(mode != Disconnected){ 
    dbg << "oport = " << oport->get_which_port() << ", ifield = " << ifield->get_which_port() << endl;
    if(is_oport){
      //
      // A new outbound connection!  Set the domain that this module is
      // associated with.
      Module *mod = oport->connection(oport->nconnections()-1)->iport->get_module();
      if(DomainManager *dm = dynamic_cast<DomainManager*>(mod)){
	domain = dm->get_domain();
      }
      else{
	// TODO: This module was connected to something other than
	// DomainManger....hmmmm....this should not be allowed
      }
    }
  }


}

} // End namespace Modules
} // End namespace PSECommon


// "We shall not cease from exploration, and the end of our
// exploring will be to arrive at the place where we started and
// to know the place for the first time."
// --T.S. Elliot
