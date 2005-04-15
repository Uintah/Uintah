// Domain Manager.cc - Provides an interface to manage a domain
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   March 2000
//
//  Copyright (C) 2000 SCI Group
//
//  Module DomainManager
//
//  Description: 
//  Input ports: 
//  Output ports:
//

#include <PSECommon/Modules/Domain/DomainManager.h>

namespace PSECommon {
namespace Modules {

extern "C" Module* make_DomainManager(const clString& id){
  return new DomainManager(id);
}

DomainManager::DomainManager(const clString& id)
  : Module("DomainManager", id, Filter),dbg("DomainManger", true),
  attribs("atttribs", id, this), geoms("geoms", id, this){
  
  domain = new Domain();
  iports.push_back(scinew FieldWrapperIPort(this, "FieldWrapper",
					    FieldWrapperIPort::Atomic));
  
  add_iport(iports[0]);

  oports.push_back(scinew DomainOPort(this, "Domain", DomainIPort::Atomic));
  add_oport(oports[0]);

}

DomainManager::~DomainManager(){
}
 
void DomainManager::execute(){
  // Check all input ports 


  // iterate over each input port, skipping the last one (which is
  // always disconnected)
  for(vector<FieldWrapperIPort*>::iterator itr = iports.begin();
      itr+1 != iports.end(); itr++){
    fwh = 0;
    (*itr)->get(fwh);
    grab(fwh);
  }
  update_lists();
  
  // Send to all output ports
  for(vector<DomainOPort*>::iterator itr = oports.begin();
      itr != oports.end(); itr++){
    (*itr)->send(domain.get_rep());
  }
  
}


void DomainManager::connection(ConnectionMode mode, int which_port,
			       int is_oport){
  if(!is_oport){
    if(mode==Disconnected){
      iports.erase(iports.begin()+which_port);
      remove_iport(which_port);
    }
    else {
      FieldWrapperIPort* p=scinew FieldWrapperIPort(this,
						    "FieldWrapper",
						    FieldWrapperIPort::Atomic);
      iports.push_back(p);
      add_iport(p);
    }
  }
  else{ // it is an outport connection
    if(mode==Disconnected){
      oports.erase(oports.begin()+which_port);
      remove_oport(which_port);
    }
    else{
      DomainOPort* p=scinew DomainOPort(this, "Domain", DomainIPort::Atomic);
      oports.push_back(p);
      add_oport(p);
    }
  }
}


// Unmarshalls the field wrapper, and stores the geom and/or attrib in the
// Domain IFF it does not allready exist in the domain.
void DomainManager::grab(const FieldWrapperHandle& ifwh){

  if(ifwh.get_rep()){
    switch(ifwh->get_field_type()){
    case GEOM:
      break;
    case ATTRIB:
      break;
    case FIELD:
      {
	FieldHandle sfh(ifwh->get_field());
	AttribHandle ah(sfh->get_attrib());
	domain->attribs[ah->getName()] = ah;
	GeomHandle gh(sfh->get_geom());
	domain->geoms[gh->get_name()] = gh;
	break;
      }
    default:
      break;
    }
  }
  else{
    return;
  }
}

void DomainManager::update_lists(){

  if(!domain->geoms.empty()){
    map<string, GeomHandle>::iterator geomitr=domain->geoms.begin();
    
    string geomlist, geominfo;
    
    if(geomitr!=domain->geoms.end()){
      geomlist = geomitr->first;
      geominfo = geomitr->second->get_info();
    }
    
    for(geomitr++; geomitr!=domain->geoms.end(); geomitr++){
      geomlist = geomlist + "," + geomitr->first;
      geominfo = geomlist + "," + geomitr->second->get_info();
    }
    dbg << "updating lists... geomlist: " << geomlist << endl;
    if(!geomlist.empty()){
      TCL::execute(id + " update_geom " + geomlist.c_str());
      TCL::execute(id + " update_geom_info " + geominfo.c_str());
    }
    
  }
  
  if(!domain->attribs.empty()){
    clString attriblist;
    map<string, AttribHandle>::iterator attribitr=domain->attribs.begin();
    
    if(attribitr!=domain->attribs.end()){
      attriblist = attribitr->first.c_str();
    }
    
    for(attribitr++; attribitr!=domain->attribs.end(); attribitr++){
      attriblist = attriblist + "," + attribitr->first.c_str();
    }
    dbg << "updating lists... attriblist: " << attriblist << endl;
    if(attriblist.len() > 0){
      TCL::execute(id+" update_attrib "+attriblist);
    }
  }
}


} // End namespace Modules
} // End namespace PSECommon

