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

class Attrib;
typedef LockingHandle<Attrib> AttribHandle;

class Attrib : public Datatype 
{
public:

  enum Type { Normal, Temp};

  // GROUP:  Constructors/Destructor
  //////////
  //
  Attrib( const string &name = string(""),  Type type = Attrib::Normal);
  virtual ~Attrib();
  
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

  //////////
  // Returns type name:
  // 0 - the class name
  // n!=0 - name of the n-th parameter (for templatazed types)
  virtual string getTypeName(int n) = 0;
 
  //////////
  // Attribute naming
  void setName(string name){
    name_=name;
  };

  string getName(){
    return name_;
  };

  Type getType() { return type_; }
  void setType( Type type=Attrib::Normal ) { type_ = type; }
  bool isTemp() { return type_ == Attrib::Temp; }

  // GROUP: Support of persistent representation
  //////////
  // Persistent IO

  void io(Piostream&);
  static  PersistentTypeID type_id;
  static  string typeName(int);

  // GROUP: Public Data
  //////////
  // 
  string      unitName_;
  
  //////////
  // Attribute creation data
  string      authorName_;
  string      date_;
  string      orgName_;
  
protected:
  string      name_;
  Type        type_;
};

}  // end namespace SCIRun

#endif
