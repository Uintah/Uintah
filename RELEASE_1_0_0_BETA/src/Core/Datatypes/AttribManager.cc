/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

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
  Pio(stream, attribHandles_);
  Pio(stream, currAttrib_);
  
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
      attribHandles_[k] = d;
    }
    
  }
				// if writing to stream
  else {
    int n = 0;
    for (iter = attribHandles_.begin(); 
	 iter != attribHandles_.end(); 
	 iter++) 
      if ( !iter->second->isTemp() )
	n++;
    
    Pio(stream, n);
    // write elements
    cerr << "attrib io " << attribHandles_.size() << "  " << n << endl;
    for (iter = attribHandles_.begin(); 
	 iter != attribHandles_.end(); 
	 iter++) {
      // have to copy iterator elements,
      // since passing them directly in a
      // call to Pio can be invalid because
      // Pio passes attribHandles_ by reference
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
  attribHandles_[empty]=AttribHandle(NULL);
  currAttrib_ = "";
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
  AttribMap::const_iterator ii=attribHandles_.find(currAttrib_);
  return (*ii).second;
}

const AttribHandle AttribManager::getAttrib(string aName) const{
  AttribMap::const_iterator ii=attribHandles_.find(aName);
 
  if (ii!=attribHandles_.end()){
    return (*ii).second;
  }
  else {
    return AttribHandle(NULL);
  }
}

void AttribManager::setCurrAttrib(string aName){
  AttribMap::const_iterator ii=attribHandles_.find(aName);
  if (ii!=attribHandles_.end()){
    currAttrib_ = aName;
  }
}

AttribHandle AttribManager::shareAttrib(string aName){
  AttribMap::const_iterator ii=attribHandles_.find(aName);
  if (ii!=attribHandles_.end()){
    return (*ii).second;
  }
  else {
    return attribHandles_[""];
  }
}


void AttribManager::addAttribute(const AttribHandle& hAttrib){
  string aName = hAttrib->getName();
  AttribMap::const_iterator ii=attribHandles_.find(aName);
  if (ii==attribHandles_.end()){
    attribHandles_[aName]=hAttrib;
    currAttrib_ = aName;
  }
}

void AttribManager::removeAttribute(string aName){
  attribHandles_.erase(aName);
}

}

