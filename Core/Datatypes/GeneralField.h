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
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/FData.h>
#include <Core/Datatypes/TypeName.h>

namespace SCIRun {
    


template <class G, class A>
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
  // If geom is set, set its name
  virtual bool set_geom_name(string iname);
  
  //////////
  // If fdata is set, set its name
  virtual bool set_fdata_name(string iname);

  //////////
  // Return geometry
  virtual const GeomHandle get_geom() const;

  //////////
  // Return the field data
  virtual const FDataHandle get_fdata() const;

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

#if 0
  //////////
  // return the min and max values
  virtual bool get_minmax(typename A::value_type &min,
			  typename A::value_type &max);
#endif


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
  static const string type_name();
  static Persistent* maker();

private:

  /////////
  // The geometry and field data associated with this field
  LockingHandle<G> geom_;
  LockingHandle<A> fdata_;
    
};

//////////
// PIO support

template <class G, class A> Persistent*
GeneralField<G,A>::maker() {
  return new GeneralField<G,A>();
}

template <class G, class A>
const string GeneralField<G,A>::type_name()
{
  static string type_name = string("GeneralField<") + find_type_name((G*)0) +","+ findTypeName((A*)0) + ">";
  return type_name;
}

template <class G, class A>
PersistentTypeID GeneralField<G,A>::type_id(GeneralField<G,A>::type_name(), 
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
GeneralField<G,A>::GeneralField(G* igeom, A* ifdata):
  Field(), geom(igeom), fdata(ifdata)
{
}

template <class G, class A >
GeneralField<G,A>::GeneralField(const GeneralField&)
{
}


template <class G, class A >
const typename A::value_type &
GeneralField<G,A>::grid(int x, int y, int z) const
{
  A* typedfdata = fdata.get_rep();
  if (typedfdata) {
    return typedfdata->get3(x, y, z);
  }
  else {
    throw "NO_FDATA_EXCEPTION";
  }
}

template <class G, class A >
typename A::value_type &
GeneralField<G,A>::operator[](int a)
{
  A* typedfdata = fdata.get_rep();
  if (typedfdata)
    {
      return typedfdata->get1(a);
    }
  else
    {
    throw "NO_FDATA_EXCEPTION";
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
bool GeneralField<G,A>::set_fdata_name(string iname){
 if (fdata.get_rep())
   {
     fdata->setName(iname);
     return true;
   }
  return false;
}

template <class G, class A >
const GeomHandle GeneralField<G,A>::get_geom() const
{
  return GeomHandle((Geom*)geom.get_rep());
}

template <class G, class A >
const FDataHandle GeneralField<G,A>::get_fdata() const
{
  return FDataHandle((FData*)fdata.get_rep());
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

#if 0
template <class T> struct MinMaxFunctor : public FdataFunctor<T>
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
  A* tfdata = fdata.get_rep();
  if (!tfdata) { return false; }

  MinMaxFunctor <typename A::value_type> f;
  switch (tfdata->dimension())
    {
    case 3:
      f.min = f.max = tfdata->get3(0, 0, 0);
      break;
    case 2:
      f.min = f.max = tfdata->get2(0, 0);
      break;
    case 1:
      f.min = f.max = tfdata->get1(0);
      break;
    default:
      return false;
    }

  tfdata->iterate(f);
  
  imin = f.min;
  imax = f.max;
  return true;
}
#endif


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
  geom->interp(fdata.get_rep(), p, outval);
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
  Pio(stream, geom_);
  Pio(stream, fdata_);
  stream.end_class();
}

} // End namespace SCIRun

#endif
