
/*
 *  Arrows.cc: Arrows object
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Geom/Arrows.h>
#include <Classlib/NotFinished.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>
#include <Malloc/Allocator.h>

GeomArrows::GeomArrows(double headwidth, double headlength)
: headwidth(headwidth), headlength(headlength)
{
    shaft_matls.add(new Material(Color(0,0,0), Color(1,0,0), Color(.6, .6, .6), 10));
    head_matls.add(new Material(Color(0,0,0), Color(0,0,1), Color(.6, .6, .6), 10));
    back_matls.add(new Material(Color(0,0,0), Color(1,0,0), Color(.6, .6, .6), 10));
}

GeomArrows::GeomArrows(const GeomArrows& copy)
: GeomObj(copy)
{
}

GeomArrows::~GeomArrows() {
}

void GeomArrows::set_matl(const MaterialHandle& shaft_matl,
			  const MaterialHandle& back_matl,
			  const MaterialHandle& head_matl)
{
    shaft_matls.resize(1);
    back_matls.resize(1);
    head_matls.resize(1);
    shaft_matls[0]=shaft_matl;
    back_matls[0]=back_matl;
    head_matls[0]=head_matl;
}

void GeomArrows::add(const Point& pos, const Vector& dir,
		     const MaterialHandle& shaft, const MaterialHandle& back,
		     const MaterialHandle& head)
{
    add(pos, dir);
    shaft_matls.add(shaft);
    back_matls.add(back);
    head_matls.add(head);
}

void GeomArrows::add(const Point& pos, const Vector& dir)
{
    positions.add(pos);
    directions.add(dir);
    if(dir.length2() < 1.e-6){
	Vector zero(0,0,0);
	v1.add(zero);
	v2.add(zero);
    } else {
	Vector vv1, vv2;
	dir.find_orthogonal(vv1, vv2);
	double len=dir.length();
	v1.add(vv1*headwidth*len);
	v2.add(vv2*headwidth*len);
    }
}

void GeomArrows::get_bounds(BBox& bb)
{
    int n=positions.size();
    for(int i=0;i<n;i++){
	bb.extend(positions[i]);
    }
}

void GeomArrows::get_bounds(BSphere& bs)
{
    int n=positions.size();
    for(int i=0;i<n;i++){
	bs.extend(positions[i]);
    }
}

void GeomArrows::make_prims(Array1<GeomObj*>&,
			  Array1<GeomObj*>&)
{
    NOT_FINISHED("GeomArrows::make_prims");
}

GeomObj* GeomArrows::clone()
{
    return scinew GeomArrows(*this);
}

void GeomArrows::preprocess()
{
    NOT_FINISHED("GeomArrows::preprocess");
}

void GeomArrows::intersect(const Ray&, Material*, Hit&)
{
    NOT_FINISHED("GeomArrows::intersect");
}

#ifdef __GNUG__

#include <Classlib/Array1.cc>
template class Array1<Point>;
template class Array1<Vector>;

#endif
