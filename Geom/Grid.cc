
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
#include <Classlib/String.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>
#include <Malloc/Allocator.h>
#include <Math/hf.h>

Persistent* make_GeomGrid()
{
    return scinew GeomGrid(0,0,Point(0,0,0), Vector(1,0,0), Vector(0,0,1),
			   GeomGrid::Regular);
}

PersistentTypeID GeomGrid::type_id("GeomGrid", "GeomObj", make_GeomGrid);

GeomGrid::GeomGrid(int nu, int nv, const Point& corner,
		   const Vector& u, const Vector& v,
		   Format format)
: nu(nu), nv(nv), corner(corner), u(u), v(v), format(format)
{
    have_matls=0;
    have_normals=0;
    switch(format){
    case Regular:
	stride=3;
	offset=0;
	break;
    case WithNormals:
	stride=6;
	offset=3;
	break;
    case WithMaterials:
	stride=7;
	offset=4;
	break;
    case WithNormAndMatl:
	stride=10;
	offset=7;
	break;
    }
    data.resize(nu*nv*stride);
    vstride=stride*nu;
    adjust();
}

GeomGrid::GeomGrid(const GeomGrid& copy)
: GeomObj(copy)
{
}

GeomGrid::~GeomGrid()
{
}

void GeomGrid::adjust()
{
    w=Cross(u, v);
    w.normalize();
    uu=u/(nu-1);
    vv=v/(nv-1);
}

void GeomGrid::get_bounds(BBox& bb)
{
    int n=nu*nv;
    float* p=&data[offset];
    float min, max;
    hf_minmax_float_s6(&data[offset], nu, nv, &min, &max);
    for(int i=0;i<8;i++){
	Point pp(corner+uu*(i&1?0:nu)+vv*(i&2?0:nv)+w*(i&4?min:max));
	bb.extend(pp);
	p+=stride;
    }
}

void GeomGrid::get_bounds(BSphere& bs)
{
    int n=nu*nv;
    float* p=&data[offset];
    for(int i=0;i<data.size();i+=stride){
	Point pp(corner+uu*p[0]+vv*p[1]+w*p[2]);
	bs.extend(pp);
	p+=stride;
    }
}

void GeomGrid::make_prims(Array1<GeomObj*>&,
			  Array1<GeomObj*>&)
{
    NOT_FINISHED("GeomGrid::make_prims");
}

GeomObj* GeomGrid::clone()
{
    return scinew GeomGrid(*this);
}

void GeomGrid::preprocess()
{
    NOT_FINISHED("GeomGrid::preprocess");
}

void GeomGrid::intersect(const Ray&, Material*, Hit&)
{
    NOT_FINISHED("GeomGrid::intersect");
}

#define GEOMGRID_VERSION 2

void GeomGrid::io(Piostream& stream)
{
    int version=stream.begin_class("GeomGrid", GEOMGRID_VERSION);
    GeomObj::io(stream);
    if(version == 1){
	cerr << "Go talk to Steve and tell him to implement this code real quick\n";
	ASSERT(0);
    }
    ASSERT(!"Not finished");
    if(stream.reading())
	adjust();
    stream.end_class();
}    

bool GeomGrid::saveobj(ostream&, const clString& format, GeomSave*)
{
    NOT_FINISHED("GeomGrid::saveobj");
    return false;
}

void GeomGrid::compute_normals()
{
    hf_float_s6(&data[0], nu, nv);
}

#ifdef __GNUG__
#include <Classlib/Array2.cc>

template class Array2<double>;
template class Array2<MaterialHandle>;
template class Array2<Vector>;

template void Pio(Piostream&, Array2<double>&);
template void Pio(Piostream&, Array2<MaterialHandle>&);
template void Pio(Piostream&, Array2<Vector>&);

#endif


#ifdef __sgi
#if _MIPS_SZPTR == 64
#include <Classlib/Array2.cc>

static void _dummy_(Piostream& p1, Array2<MaterialHandle>& p2)
{
    Pio(p1, p2);
}

static void _dummy_(Piostream& p1, Array2<Vector>& p2)
{
    Pio(p1, p2);
}

static void _dummy_(Piostream& p1, Array2<double>& p2)
{
    Pio(p1, p2);
}

#endif
#endif

