// Attrib.h - the base attribute class.
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute
//  General storage class for Fields.

#ifndef SCI_project_Attrib_h
#define SCI_project_Attrib_h 1

#include <vector>
#include <string>

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Exceptions/DimensionMismatch.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Util/FancyAssert.h>


namespace SCIRun {

using namespace std;

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
  

  //virtual void get1(T &result, int x) = 0;
  //virtual void get2(T &result, int x, int y) = 0;
  //virtual void get3(T &result, int x, int y, int z) = 0;


  /////////
  // Cast down to specific type.
  template <class A> A *downcast(A *) { return dynamic_cast<A *>(this); }
  
  /////////
  // set (and get) the name of the attribute
  void setName(std::string iname) {d_name = iname; };
  std::string getName() { return d_name; };

  /////////
  // Get information about the attribute
  virtual string getInfo() =0;
  
protected:
  std::string d_name;
};

} // End namespace SCIRun

#endif



