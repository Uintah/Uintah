// Attrib.h - the base attribute class.
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute
//
//  General storage class for Fields.
//

#ifndef SCI_project_Attrib_h
#define SCI_project_Attrib_h 1

#include <vector>
#include <string>

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Exceptions/DimensionMismatch.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>


namespace SCICore {
namespace Datatypes {

using namespace std;
using SCICore::Containers::LockingHandle;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

template <class T> class FlatAttrib;
class Attrib;
typedef LockingHandle<Attrib> AttribHandle;

class Attrib : public Datatype 
{
public:
  //////////
  // Constructors, Destructor
  Attrib() {};
  Attrib(const Attrib &) {};

  virtual ~Attrib() { };
  
  /////////
  // Cast down to specific type.
  template <class A> A *downcast(A *);
  
  /////////
  // set (and get) the name of the attribute
  void set_name(std::string iname){name=iname; };
  std::string get_name(){return name;};

  /////////
  //
  virtual string get_info() =0;
  
protected:
  std::string name;
};

template <class A> A *
Attrib::downcast(A *)
{
  return dynamic_cast<A *>(this);
}


}  // end Datatypes
}  // end SCICore

#endif



