//  GeneralField.h - A general scalar field, comprised of one attribute and one geometry
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute
//  DESCRIPTION:
//  This class represents a basic scalar field, containing one
//  attribute and one geometry.  The attribute (template argument A)
//  defaults to a DiscreteSAttrib unless otherwise specified at compile
//  time.

#ifndef SCI_project_GeneralField_h
#define SCI_project_GeneralField_h 1


#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/SField.h>
#include <Core/Datatypes/DiscreteAttrib.h>
#include <Core/Datatypes/TypeName.h>

namespace SCIRun {
    


template <class G, class A=DiscreteAttrib<double> >
  class SCICORESHARE GeneralField : public Field
{
  public:

  /////////
  // Constructors
  GeneralField();
  GeneralField(G*, A*);
  GeneralField(const GeneralField&);

  /////////
  // Destructors
  ~GeneralField();

  virtual const typename A::value_type &grid(int, int, int) const;
  virtual typename A::value_type &operator[](int);

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
  virtual const GeomHandle getGeom() const;

  //////////
  // Return the attribute
  virtual const AttribHandle getAttrib() const;

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
  // return the min and max values
  virtual bool get_minmax(typename A::value_type &min,
			  typename A::value_type &max);

  //////////
  // Walk the field, applying op to each node
  //bool walk(itr, const BBox&, BinaryFunction op)
  
  //////////
  // Interpolate at the point
  virtual bool interpolate(const Point& ipoint,
			   typename A::value_type &outval,
			   double eps=1.e-6);

  /////////
  // Compute the gradient at the point
  virtual Vector gradient(const Point&);
  
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static string typeName();
  static Persistent* maker();

private:

  /////////
  // The geometry and attribute associated with this field
  LockingHandle<G> geom;
  LockingHandle<A> attrib;
    
};

//////////
// PIO support

template <class G, class A> Persistent*
GeneralField<G,A>::maker() {
  return new GeneralField<G,A>();
}

template <class G, class A>
string GeneralField<G,A>::typeName() {
  static string typeName = string("GeneralField<") + findTypeName((G*)0) +","+ findTypeName((A*)0) + ">";
  return typeName;
}

template <class G, class A>
PersistentTypeID GeneralField<G,A>::type_id(GeneralField<G,A>::typeName(), 
					 "Field", 
					 GeneralField<G,A>::maker);

template <class G, class A >
GeneralField<G,A>::GeneralField():
  Field()
{
}

template <class G, class A >
GeneralField<G,A>::~GeneralField()
{
}

template <class G, class A >
GeneralField<G,A>::GeneralField(G* igeom, A* iattrib):
  Field(), geom(igeom), attrib(iattrib)
{
}

template <class G, class A >
GeneralField<G,A>::GeneralField(const GeneralField&)
{
}


template <class G, class A >
bool
GeneralField<G, A>::resize(int x, int y, int z)
{
  A *typedattrib = attrib.get_rep();
  if (typedattrib) { typedattrib->resize(x, y, z); }
  
  G *typedgeom = geom.get_rep();
  if (typedgeom) { typedgeom->resize(x, y, z); }

  return true;
}



template <class G, class A >
const typename A::value_type &
GeneralField<G,A>::grid(int x, int y, int z) const
{
  A* typedattrib = attrib.get_rep();
  if (typedattrib) {
    return typedattrib->get3(x, y, z);
  }
  else {
    throw "NO_ATTRIB_EXCEPTION";
  }
}

template <class G, class A >
typename A::value_type &
GeneralField<G,A>::operator[](int a)
{
  A* typedattrib = attrib.get_rep();
  if (typedattrib)
    {
      return typedattrib->get1(a);
    }
  else
    {
    throw "NO_ATTRIB_EXCEPTION";
    }
}

template <class G, class A >
bool GeneralField<G,A>::set_geom_name(string iname)
{
  if (geom.get_rep())
    {
      geom->setName(iname);
      return true;
    }
  return false;
}

template <class G, class A >
bool GeneralField<G,A>::set_attrib_name(string iname){
 if (attrib.get_rep())
   {
     attrib->setName(iname);
     return true;
   }
  return false;
}

template <class G, class A >
const GeomHandle GeneralField<G,A>::getGeom() const
{
  return GeomHandle((Geom*)geom.get_rep());
}

template <class G, class A >
const AttribHandle GeneralField<G,A>::getAttrib() const
{
  return AttribHandle((Attrib*)attrib.get_rep());
}

template <class G, class A >
bool GeneralField<G,A>::get_bbox(BBox& bbox)
{
  if(geom.get_rep())
    {
      if(geom->getBoundingBox(bbox))
	{
	  return true;
	}
      else
	{
	  return false;
	}
    }
  else
    {
      return false;
    }
}

template <class T> struct MinMaxFunctor : public AttribFunctor<T>
{
public:
  virtual void operator () (T &val)
  {
    min = Min(val, min);
    max = Max(val, max);
  }

  T min;
  T max;
};


// TODO: Implement this so it's always valid.
template <class G, class A >
bool GeneralField<G,A>::get_minmax(typename A::value_type &imin,
				   typename A::value_type &imax)
{
  A* tattrib = attrib.get_rep();
  if (!tattrib) { return false; }

  MinMaxFunctor <typename A::value_type> f;
  switch (tattrib->dimension())
    {
    case 3:
      f.min = f.max = tattrib->get3(0, 0, 0);
      break;
    case 2:
      f.min = f.max = tattrib->get2(0, 0);
      break;
    case 1:
      f.min = f.max = tattrib->get1(0);
      break;
    default:
      return false;
    }

  tattrib->iterate(f);
  
  imin = f.min;
  imax = f.max;
  return true;
}


template <class G, class A >
bool GeneralField<G,A>::longest_dimension(double &odouble)
{
  if(geom.get_rep()) {
    return geom->longestDimension(odouble);
  }
  else{
    return false;
  }
}

//template <class T, class G, class A >
//BinaryFunction GeneralField<T,G,A>::walk(const BBox& ibbox, BinaryFunction op){
//  // foreach node inside ibbox
//  op(thisnode);
//  // return the BinaryFunction
//  return op;
//}

template <class G, class A >
bool GeneralField<G,A>::interpolate(const Point& p,
				    typename A::value_type &outval,
				    double)
{
  geom->interp(attrib.get_rep(), p, outval);
  return true;
}

template <class G, class A >
Vector GeneralField<G,A>::gradient(const Point& /* ipoint */)
{
  return Vector();
}

#define GeneralField_VERSION 1

template <class G, class A >
void GeneralField<G,A>::io(Piostream& stream){

  stream.begin_class(typeName().c_str(), GeneralField_VERSION);
  //  Field::io(stream);
  Pio(stream, geom);
  Pio(stream, attrib);
  stream.end_class();
}

} // End namespace SCIRun

#endif
