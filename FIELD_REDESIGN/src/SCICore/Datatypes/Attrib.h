// Attrib.h - the base attribute class.
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_Attrib_h
#define SCI_project_Attrib_h 1

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>

#include <vector>
#include <string>

namespace SCICore{
namespace Datatypes{

using std::vector;
using std::string;
using SCICore::Containers::LockingHandle;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class Attrib;
typedef LockingHandle<Attrib> AttribHandle;

class Attrib:public Datatype 
{
public:
  //////////
  // Destructor
  virtual ~Attrib() { };

  /////////
  // set (and get) the name of the attribute
  void set_name(string iname){name=iname; };
  string get_name(){return name;};
  
protected:
  /////////
  // an identifying name
  string name;
};



}  // end Datatypes
}  // end SCICore

#endif



