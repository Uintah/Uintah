
/*
 *  Grid.cc: Grid object
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Geom/Grid.h>
#include <Classlib/NotFinished.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>

GeomGrid::GeomGrid(int nu, int nv, const Point& corner,
		   const Vector& u, const Vector& v)
: verts(nu, nv), corner(corner), u(u), v(v)
{
    have_matls=0;
    have_normals=0;
    w=Cross(u, v);
    w.normalize();
}

GeomGrid::GeomGrid(const GeomGrid& copy)
: GeomObj(copy)
{
}

GeomGrid::~GeomGrid() {
}

void GeomGrid::set(int i, int j, double v)
{
    verts(i, j)=v;
}

void GeomGrid::set(int i, int j, double v, const Vector& normal)
{
    if(!have_normals){
	normals.newsize(verts.dim1(), verts.dim2());
	have_normals=1;
    }
    verts(i, j)=v;
    normals(i, j)=normal;
}

void GeomGrid::set(int i, int j, double v, const MaterialHandle& matl)
{
    if(!have_matls){
	matls.newsize(verts.dim1(), verts.dim2());
	have_matls=1;
    }
    verts(i, j)=v;
    matls(i, j)=matl;
}

void GeomGrid::set(int i, int j, double v, const Vector& normal,
		   const MaterialHandle& matl)
{
    if(!have_matls){
	matls.newsize(verts.dim1(), verts.dim2());
	have_matls=1;
    }
    if(!have_normals){
	normals.newsize(verts.dim1(), verts.dim2());
	have_normals=1;
    }
    verts(i, j)=v;
    matls(i, j)=matl;
    normals(i, j)=normal;
}

void GeomGrid::get_bounds(BBox& bb)
{
    int nu=verts.dim1();
    int nv=verts.dim2();
    Vector uu(u/(nu-1));
    Vector vv(v/(nv-1));
    Point rstart(corner);
    for(int i=0;i<nu;i++){
	Point p(rstart);
	for(int j=0;j<nv;j++){
	    Point pp(p+w*verts(i, j));
	    bb.extend(pp);
	    p+=uu;
	}
	rstart+=vv;
    }
}

void GeomGrid::get_bounds(BSphere& bs)
{
    int nu=verts.dim1();
    int nv=verts.dim2();
    Vector uu(u/(nu-1));
    Vector vv(v/(nv-1));
    Point rstart(corner);
    for(int i=0;i<nu;i++){
	Point p(rstart);
	for(int j=0;j<nv;j++){
	    Point pp(p+w*verts(i, j));
	    bs.extend(pp);
	    p+=uu;
	}
	rstart+=vv;
    }
}

void GeomGrid::make_prims(Array1<GeomObj*>&,
			  Array1<GeomObj*>&)
{
    NOT_FINISHED("GeomGrid::make_prims");
}

GeomObj* GeomGrid::clone()
{
    return new GeomGrid(*this);
}

void GeomGrid::preprocess()
{
    NOT_FINISHED("GeomGrid::preprocess");
}

void GeomGrid::intersect(const Ray&, Material*, Hit&)
{
    NOT_FINISHED("GeomGrid::intersect");
}
