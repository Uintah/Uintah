//  AttribManager.cc - Manages a collection of Attributes
//
//  Written by:
//   Yarden Livnat
//   Department of Computer Science
//   University of Utah
//   Jan. 2001
//
//  Copyright (C) 2001 SCI Institute

#include <Core/Datatypes/AttribManager.h>
#include <Core/Persistent/PersistentSTL.h>

namespace SCIRun{

using std::cout;
using std::endl;

// GROUP: Persistence support
//////////
//

static Persistent* maker(){
  return new AttribManager();
}

PersistentTypeID AttribManager::type_id("AttribManager", "Datatype", maker);

//////////
// Persistent IO
#define ATTRIB_MANAGER_VERSION 1

void AttribManager::io(Piostream& stream){
  
  stream.begin_class("AttribManager", ATTRIB_MANAGER_VERSION);
  Pio(stream, d_attribHandles);
  Pio(stream, d_currAttrib);
  
  stream.end_class();
}

void
AttribManager::io_map( Piostream& stream )
{
  AttribMap::iterator iter;
  int i, n;
  string k;
  AttribHandle d;
  
  if (stream.reading()) {	
				// get map size 
    Pio(stream, n);
				// read elements
    for (i = 0; i < n; i++) {
      Pio(stream, k);
      Pio(stream, d);
      d_attribHandles[k] = d;
    }
    
  }
				// if writing to stream
  else {
    int n = 0;
    for (iter = d_attribHandles.begin(); 
	 iter != d_attribHandles.end(); 
	 iter++) 
      if ( !iter->second->isTemp() )
	n++;
    
    Pio(stream, n);
    // write elements
    cerr << "attrib io " << d_attribHandles.size() << "  " << n << endl;
    for (iter = d_attribHandles.begin(); 
	 iter != d_attribHandles.end(); 
	 iter++) {
      // have to copy iterator elements,
      // since passing them directly in a
      // call to Pio can be invalid because
      // Pio passes d_attribHandles by reference
      if ( !iter->second->isTemp() ) {
	string ik = (*iter).first;
	AttribHandle dk = (*iter).second;
	Pio(stream, ik);
	Pio(stream, dk);
      }
      else 
	cerr << "Skipping temp Atrribute:" << iter->first << "\n";
    }
    
  }
}

//////////
// Constructors/Destructor
AttribManager::AttribManager()
{
  string empty("");
  d_attribHandles[empty]=AttribHandle(NULL);
  d_currAttrib = "";
}

AttribManager::AttribManager(const AttribManager&)
{
  // TODO: implement this!!!
}

AttribManager::~AttribManager(){
}

//////////
// Member functions implementation
const AttribHandle AttribManager::getAttrib() const{
  AttribMap::const_iterator ii=d_attribHandles.find(d_currAttrib);
  return (*ii).second;
}

const AttribHandle AttribManager::getAttrib(string aName) const{
  AttribMap::const_iterator ii=d_attribHandles.find(aName);
 
  if (ii!=d_attribHandles.end()){
    return (*ii).second;
  }
  else {
    return AttribHandle(NULL);
  }
}

void AttribManager::setCurrAttrib(string aName){
  AttribMap::const_iterator ii=d_attribHandles.find(aName);
  if (ii!=d_attribHandles.end()){
    d_currAttrib = aName;
  }
}

AttribHandle AttribManager::shareAttrib(string aName){
  AttribMap::const_iterator ii=d_attribHandles.find(aName);
  if (ii!=d_attribHandles.end()){
    return (*ii).second;
  }
  else {
    return d_attribHandles[""];
  }
}


void AttribManager::addAttribute(const AttribHandle& hAttrib){
  string aName = hAttrib->getName();
  AttribMap::const_iterator ii=d_attribHandles.find(aName);
  if (ii==d_attribHandles.end()){
    d_attribHandles[aName]=hAttrib;
    d_currAttrib = aName;
  }
}

void AttribManager::removeAttribute(string aName){
  d_attribHandles.erase(aName);
}

}

