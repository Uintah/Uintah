//  Field.cc - This is the base class from which all other fields are derived.
//
//  Written by:
//   Eric Kuehne, Alexei Samsonov
//   Department of Computer Science
//   University of Utah
//   April 2000, December 2000
//
//  Copyright (C) 2000 SCI Institute

#include <Core/Datatypes/Field.h>
#include <Core/Persistent/PersistentSTL.h>

namespace SCIRun{

using std::cout;
using std::endl;

// GROUP: Persistence support
//////////
//

static Persistent* maker(){
  return new Field();
}

PersistentTypeID Field::type_id("Field", "Datatype", maker);

//////////
// Persistent IO
#define FIELD_VERSION 1

void Field::io(Piostream& stream){
  
  stream.begin_class("Field", FIELD_VERSION);
  Pio(stream, d_attribHandles);
  Pio(stream, d_geomHandle);
  Pio(stream, d_currAttrib);
  
  stream.end_class();
}


//////////
// Constructors/Destructor
Field::Field()
{
  string empty("");
  d_attribHandles[empty]=AttribHandle(NULL);
  d_currAttrib = "";
}

Field::Field(const Field&)
{
  // TODO: implement this!!!
}

Field::~Field(){
}

//////////
// Member functions implementation
const AttribHandle Field::getAttrib() const{
  AttribMap::const_iterator ii=d_attribHandles.find(d_currAttrib);
  return (*ii).second;
}

const AttribHandle Field::getAttrib(string aName) const{
  AttribMap::const_iterator ii=d_attribHandles.find(aName);
 
  if (ii!=d_attribHandles.end()){
    return (*ii).second;
  }
  else {
    return AttribHandle(NULL);
  }
}

void Field::setCurrAttrib(string aName){
  AttribMap::const_iterator ii=d_attribHandles.find(aName);
  if (ii!=d_attribHandles.end()){
    d_currAttrib = aName;
  }
}

AttribHandle Field::shareAttrib(string aName){
  AttribMap::const_iterator ii=d_attribHandles.find(aName);
  if (ii!=d_attribHandles.end()){
    return (*ii).second;
  }
  else {
    return d_attribHandles[""];
  }
}

const GeomHandle Field::getGeom() const{
  return d_geomHandle;
}

void Field::addAttribute(const AttribHandle& hAttrib){
  string aName = hAttrib->getName();
  AttribMap::const_iterator ii=d_attribHandles.find(aName);
  if (ii==d_attribHandles.end()){
    d_attribHandles[aName]=hAttrib;
  }
}

void Field::removeAttribute(string aName){
  d_attribHandles.erase(aName);
}

bool  Field::setGeometry(GeomHandle hGeom){
  d_geomHandle=hGeom;
  // TODO: checking with consistency with existin attributes???
  return true;
}

}

