// Attrib.cc - the base attribute class.
//
//  Written by:
//   Eric Kuehne, Alexei Samsonov
//   Department of Computer Science
//   University of Utah
//   April 2000, December 2000
//
//  Copyright (C) 2000 SCI Institute

#include <Core/Datatypes/Attrib.h>
#include <Core/Exceptions/DimensionMismatch.h>

namespace SCIRun {

using std::cout;
using std::endl;

// GROUP: Implementation of Attrib class
//////////
//

//////////
// PIO support
string Attrib::typeName(int){
  static string name = "Attrib";
  return name;
}

PersistentTypeID Attrib::type_id(typeName(0), 
				 "Datatype", 
				 0);
#define ATTRIB_VERSION 1

void Attrib::io(Piostream& stream){
  
  stream.begin_class(typeName(0).c_str(), ATTRIB_VERSION);
  
  cout << "Starting attrib output" << endl;
  Pio(stream, d_name);
  Pio(stream, d_authorName);
  Pio(stream, d_date);
  Pio(stream, d_orgName);
  Pio(stream, d_unitName);
  
  stream.end_class();
}

//////////
// Constructor/Destructor
Attrib::Attrib( const string &name, Type type) : d_name(name), d_type(type) {
}


Attrib::~Attrib(){
}

string Attrib::getTypeName(int n){
  return typeName(n);
}

}  // end namespace SCIRun
