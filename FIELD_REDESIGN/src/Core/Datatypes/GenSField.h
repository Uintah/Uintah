//  GenSField.h - A general scalar field, comprised of one attribute and one geometry
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_GenSField_h
#define SCI_project_GenSField_h 1


#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Datatypes/SField.h>

namespace SCICore{
namespace Datatypes{

using SCICore::Containers::LockingHandle;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;


template <class T> class SCICORESHARE GenSField:
  public SField,
  public SInterpolate<T>{
public:

  /////////
  // Constructors
  GenSField();
  GenSField(Geom*, Attrib*);
  GenSField(const GenSField&);

  /////////
  // Destructors
  ~GenSField();


  virtual T& grid(const Point&);
  virtual T& operator[](int);
  virtual void resize(int, int, int);
  virtual void set_bounds(const Point&, const Point&);

  //////////
  // Test to see if this field includes (is derived from)
  // the given interface. As convention, the input string should be in
  // all lowercase and be exactly the same as the FieldInterface's
  // name.  For example:
  //
  // SInterpolate *inter = some_field.query_interface("sinterpolate");
  //
  // Returns NULL if the given interface is not available for the field.
  virtual FieldInterface* query_interface(const string&);

  virtual int sinterpolate(const Point& ipoint, T& outval);
    
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

private:
  
};


} // end SCICore
} // end Datatypes

#endif
