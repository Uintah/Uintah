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

PersistentTypeID Attrib::type_id("Attrib", "Datatype", 0);

//////////
// Persistent IO
#define ATTRIB_VERSION 1

void Attrib::io(Piostream& stream){
  
  stream.begin_class("Attrib", ATTRIB_VERSION);
  
  cout << "Starting attrib output" << endl;
  Pio(stream, d_name);
  Pio(stream, d_authorName);
  Pio(stream, d_date);
  Pio(stream, d_orgName);
  Pio(stream, d_unitName);

  stream.end_class();
}

}  // end namespace SCIRun
