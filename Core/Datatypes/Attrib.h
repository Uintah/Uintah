// Attrib.h - the base attribute class.
//
//  Written by:
//   Eric Kuehne, Alexei Samsonov
//   Department of Computer Science
//   University of Utah
//   April 2000, December 2000
//
//  Copyright (C) 2000 SCI Institute
//
//  General storage class for Fields.
//

#ifndef SCI_project_Attrib_h
#define SCI_project_Attrib_h 1

#include <vector>
#include <string>
#include <iostream>

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Exceptions/DimensionMismatch.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Datatypes/TypeName.h>
#include <Core/Persistent/PersistentSTL.h>

namespace SCIRun {

using namespace std;

/////////
// Structure to hold Newmann BC related values
class NewmannBC {
public:  
  // GROUP: public data
  //////////
  // 
  NewmannBC(){};
  NewmannBC(Vector v, double d): dir(v), val(d){};
  //////////
  // Direction to take derivative in
  Vector dir;
  //////////
  // Value of the derivative
  double val;
};

//////////
// PIO for NewmannBC objects
void  Pio(Piostream&, NewmannBC&);
ostream& operator<<(ostream&, NewmannBC&);

class Attrib;
typedef LockingHandle<Attrib> AttribHandle;

class Attrib : public Datatype 
{
public:

  // GROUP:  Constructors/Destructor
  //////////
  //
  Attrib(){};
  virtual ~Attrib() { };
  
  // GROUP: Class interface functions
  //////////
  // 
  
  /////////
  // Casts down to handle to attribute of specific type.
  // Returns empty handle if it was not successeful cast
  template <class T> LockingHandle<T> downcast(T*) {
    T* rep = dynamic_cast<T*>(this);
    return LockingHandle<T>(rep);
  }

  //////////
  // Get information about the attribute
  virtual string getInfo() = 0;

  void setName(string name){
    d_name=name;
  };

  string getName(){
    return d_name;
  };

  // GROUP: Support of persistent representation
  //////////
  // Persistent IO
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  
  // GROUP: Public Data
  //////////
  // 
  string      d_unitName;
  
  //////////
  // Attribute creation data
  string      d_authorName;
  string      d_date;
  string      d_orgName;
  
 
protected:
  string      d_name;
};

}  // end namespace SCIRun

#endif
