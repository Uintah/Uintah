//  FlatSAttrib.h - scalar attribute stored as a flat array
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#ifndef SCI_project_FlatSAttrib_h
#define SCI_project_FlatSAttrib_h 1

#include <SCICore/Datatypes/SAttrib.h>
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

template <class T> class SCICORESHARE FlatSAttrib:public SAttrib 
{
public:

  //////////
  // Constructor
  FlatSAttrib();

  //////////
  // Destructor
  ~FlatSAttrib();

  //////////
  // return the value at the given position
  T operator [](int);

  //////////
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

private:
  vector<T> data;
  
};



}  // end Datatypes
}  // end SCICore

#endif



