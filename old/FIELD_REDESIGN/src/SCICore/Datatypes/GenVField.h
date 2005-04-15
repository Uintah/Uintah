//  GenVField.h - A general scalar field, comprised of one attribute and one geometry
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
//  defaults to a DiscreteSAttrib unless otherwise specified at compile
//  time.
//
//

#ifndef SCI_project_GenVField_h
#define SCI_project_GenVField_h 1


#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Datatypes/VField.h>
#include <SCICore/Datatypes/DiscreteAttrib.h>

namespace SCICore{
namespace Datatypes{
    
using SCICore::Containers::LockingHandle;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;


template <class T, class G, class A=DiscreteAttrib<T> > 
  class SCICORESHARE GenVField: public VField, public VLInterpolate
{
  public:
    
  /////////
  // Constructors
  GenVField();
  GenVField(G*, A*);
  GenVField(const GenVField&);

  /////////
  // Destructors
  ~GenVField();

  virtual const T& grid(int, int, int) const;
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
  //virtual bool set_bbox(const BBox&);
  //virtual bool set_bbox(const Point&, const Point&);

  //////////
  // Walk the field, applying op to each node
  //bool walk(itr, const BBox&, BinaryFunction op)
  
  //////////
  // Interpolate at the point
  virtual int vlinterpolate(const Point& ipoint, Vector& outval);

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

template <class T, class G, class A>
PersistentTypeID GenVField<T,G,A>::type_id("GenVField", "Datatype", 0);

template <class T, class G, class A >
GenVField<T,G,A>::GenVField():
  VField()
{
}

template <class T, class G, class A >
GenVField<T,G,A>::~GenVField()
{
}

template <class T, class G, class A >
GenVField<T,G,A>::GenVField(G* igeom, A* iattrib):
  VField(), geom(igeom), attrib(iattrib)
{
}

template <class T, class G, class A >
GenVField<T,G,A>::GenVField(const GenVField&)
{
}


template <class T, class G, class A >
bool
GenVField<T, G, A>::resize(int x, int y, int z)
{
  A *typedattrib = attrib.get_rep();
  if (typedattrib) { typedattrib->resize(x, y, z); }
  
  G *typedgeom = geom.get_rep();
  if (typedgeom) { typedgeom->resize(x, y, z); }

  return true;
}



template <class T, class G, class A >
const T& GenVField<T,G,A>::grid(int x, int y, int z) const
{
  A* typedattrib = attrib.get_rep();
  if (typedattrib) {
    return typedattrib->get3(x, y, z);
  }
  else {
    throw "NO_ATTRIB_EXCEPTION";
  }
}

template <class T, class G, class A >
T& GenVField<T,G,A>::operator[](int a)
{
  A* typedattrib = attrib.get_rep();
  if (typedattrib)
    {
      return (T &)(typedattrib->get1(a));
    }
  else
    {
    throw "NO_ATTRIB_EXCEPTION";
    }
}

template <class T, class G, class A >
bool GenVField<T,G,A>::set_geom_name(string iname)
{
  if (geom.get_rep())
    {
      geom->setName(iname);
      return true;
    }
  return false;
}

template <class T, class G, class A >
bool GenVField<T,G,A>::set_attrib_name(string iname){
 if (attrib.get_rep())
   {
     attrib->setName(iname);
     return true;
   }
  return false;
}

template <class T, class G, class A >
Geom* GenVField<T,G,A>::get_geom()
{
  return (Geom*) geom.get_rep();
}

template <class T, class G, class A >
Attrib* GenVField<T,G,A>::get_attrib()
{
  return attrib.get_rep();
}

template <class T, class G, class A >
bool GenVField<T,G,A>::get_bbox(BBox& bbox)
{
  if(geom.get_rep())
    {
      if(geom->getBoundingBox(bbox))
	{
	  return 1;
	}
      else
	{
	  return 0;
	}
    }
  else
    {
      return 0;
    }
}


template <class T, class G, class A >
bool GenVField<T,G,A>::longest_dimension(double &odouble)
{
  if(geom.get_rep()) {
    return geom->longestDimension(odouble);
  }
  else{
    return false;
  }
}

//template <class T, class G, class A >
//BinaryFunction GenVField<T,G,A>::walk(const BBox& ibbox, BinaryFunction op){
//  // foreach node inside ibbox
//  op(thisnode);
//
//  // return the BinaryFunction
//  return op;
//}


template <class T, class G, class A >
int GenVField<T,G,A>::vlinterpolate(const Point& p, Vector& outval)
{
  T out;
  geom->interp(attrib.get_rep(), p, out);
  outval = out;
  return 0;
}


template <class T, class G, class A >
Vector GenVField<T,G,A>::gradient(const Point& /* ipoint */)
{
  return Vector();
}


template <class T, class G, class A >
void GenVField<T,G,A>::io(Piostream&){
}

} // end SCICore
} // end Datatypes

#endif
