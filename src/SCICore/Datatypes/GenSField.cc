//  GenSField.cc - A general scalar field, comprised of one attribute and one geometry
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/GenSField.h>

namespace SCICore{
namespace Datatypes{

template <class T, class G, class A>
PersistentTypeID GenSField<T,G,A>::type_id("GenSField", "Datatype", 0);

template <class T, class G, class A >
GenSField<T,G,A>::GenSField():
  SField()
{
}

template <class T, class G, class A >
GenSField<T,G,A>::~GenSField()
{
}

template <class T, class G, class A >
GenSField<T,G,A>::GenSField(G* igeom, A* iattrib):
  SField(), geom(igeom), attrib(iattrib)
{
}

template <class T, class G, class A >
GenSField<T,G,A>::GenSField(const GenSField&)
{
}

template <class T, class G, class A >
const T& GenSField<T,G,A>::grid(int x, int y, int z) const
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
T& GenSField<T,G,A>::operator[](int a)
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
bool GenSField<T,G,A>::set_geom_name(string iname)
{
  if (geom.get_rep())
    {
      geom->set_name(iname);
      return true;
    }
  return false;
}

template <class T, class G, class A >
bool GenSField<T,G,A>::set_attrib_name(string iname){
 if (attrib.get_rep())
   {
     attrib->set_name(iname);
     return true;
   }
  return false;
}

template <class T, class G, class A >
Geom* GenSField<T,G,A>::get_geom()
{
  return (Geom*) geom.get_rep();
}

template <class T, class G, class A >
Attrib* GenSField<T,G,A>::get_attrib()
{
  return attrib.get_rep();
}

template <class T, class G, class A >
bool GenSField<T,G,A>::get_bbox(BBox& bbox)
{
  if(geom.get_rep())
    {
      if(geom->get_bbox(bbox))
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

template <class T> struct MinMaxFunctor : public AttribFunctor<T>
{
public:
  virtual void operator () (T &val)
  {
    if (val < min) { min = val; }
    if (val > max) { max = val; }
  }

  T min;
  T max;
};


// TODO: Implement this so it's always valid.
template <class T, class G, class A >
bool GenSField<T,G,A>::get_minmax(double &imin, double &imax)
{
  A* tattrib = attrib.get_rep();
  if (!tattrib) { return false; }

  MinMaxFunctor<T> f;
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
  
  imin = (double)f.min;
  imax = (double)f.max;
  return true;
}


template <class T, class G, class A >
bool GenSField<T,G,A>::longest_dimension(double &odouble)
{
  if(geom.get_rep()) {
    return geom->longest_dimension(odouble);
  }
  else{
    return false;
  }
}

//template <class T, class G, class A >
//BinaryFunction GenSField<T,G,A>::walk(const BBox& ibbox, BinaryFunction op){
//  // foreach node inside ibbox
//  op(thisnode);
//
//  // return the BinaryFunction
//  return op;
//}

template <class T, class G, class A >
int GenSField<T,G,A>::slinterpolate(const Point& p, double& outval, double eps)
{
  return geom->slinterpolate<A>(attrib.get_rep(), data_loc, p, outval, eps);
}

template <class T, class G, class A >
Vector GenSField<T,G,A>::gradient(const Point& /* ipoint */)
{
  return Vector();
}


template <class T, class G, class A >
void GenSField<T,G,A>::io(Piostream&){
}

} // end Datatypes
} // end SCICore

