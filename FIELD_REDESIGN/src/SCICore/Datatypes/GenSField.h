//  GenSField.h - A general scalar field, comprised of one attribute and one geometry
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute
//
//  DESCRIPTION:
//
//  This class represents a basic scalar field, containing one
//  attribute and one geometry.  The attribute (template argument A)
//  defaults to a FlatSAttrib unless otherwise specified at compile
//  time.
//
//

#ifndef SCI_project_GenSField_h
#define SCI_project_GenSField_h 1


#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Datatypes/SField.h>
#include <SCICore/Datatypes/FlatAttrib.h>

namespace SCICore{
namespace Datatypes{
    
using SCICore::Containers::LockingHandle;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;


template <class T, class G, class A=FlatAttrib<T> > class SCICORESHARE GenSField:
public SField,
public SLInterpolate{
public:
    
  /////////
  // Constructors
  GenSField();
  GenSField(G*, A*);
  GenSField(const GenSField&);

  /////////
  // Destructors
  ~GenSField();

  virtual T& grid(int, int, int);
  virtual T& operator[](int);

  //////////
  // Resize the geom and return true if it is of type LatticeGeom,
  // return false if not.
  virtual bool resize(int, int, int);
  
  //////////
  // If geom is set, set its name
  virtual bool set_geom_name(string iname);
  
  //////////
  // If attrib is set, set its name
  virtual bool set_attrib_name(string iname);

  //////////
  // Return geometry
  virtual Geom* get_geom();

  //////////
  // Return the attribute
  virtual Attrib* get_attrib();

  //////////
  // Return the upper and lower bounds
  virtual bool get_bbox(BBox&);

  //////////
  // return the longest dimension of the field
  virtual bool longest_dimension(double&);
  
  //////////
  // Set the bounding box
  virtual bool set_bbox(const BBox&);
  virtual bool set_bbox(const Point&, const Point&);

  //////////
  // return the min and max values
  virtual bool get_minmax(double&, double&);

  //////////
  // Walk the field, applying op to each node
  //bool walk(itr, const BBox&, BinaryFunction op)
  
  //////////
  // Interpolate at the point
  virtual int slinterpolate(const Point& ipoint, double& outval, double eps=1.e-6);

  /////////
  // Compute the gradient at the point
  virtual Vector gradient(const Point&);
  
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

private:

  /////////
  // The geometry and attribute associated with this field
  LockingHandle<G> geom;
  LockingHandle<A> attrib;
    
};


} // end SCICore
} // end Datatypes

#endif
