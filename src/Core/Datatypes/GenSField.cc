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
bool GenSField<T,G,A>::resize(int a, int b, int c)
{
  LatticeGeom* lgeom = geom->get_latticegeom();
  if (lgeom && attrib.get_rep()) {
    lgeom->resize(a, b, c);
    attrib->resize(a, b, c);
    return true;
  }
  else{
    return false;
  }
}

template <class T, class G, class A >
T& GenSField<T,G,A>::grid(int x, int y, int z)
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
      return typedattrib->get1(a);
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


template <class T, class G, class A >
bool GenSField<T,G,A>::set_bbox(const BBox& bbox)
{
  if (geom.get_rep())
    {
      if(geom->set_bbox(bbox))
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
bool GenSField<T,G,A>::set_bbox(const Point& p1, const Point& p2)
{
  if (geom.get_rep())
    {
      if(geom->set_bbox(p1, p2))
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

// TODO: Implement this so it's always valid.
template <class T, class G, class A >
bool GenSField<T,G,A>::get_minmax(double& /* imin */, double& /* imax */)
{
  return false;
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

