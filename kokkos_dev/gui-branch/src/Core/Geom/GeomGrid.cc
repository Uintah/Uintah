/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  GeomGrid.cc: Grid object
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifdef BROKEN

#include <Core/Geom/GeomGrid.h>
#include <Core/Util/NotFinished.h>
#include <Core/Containers/String.h>
#include <Core/Geom/Save.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/BSphere.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/hf.h>

namespace SCIRun {

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
    if(version == 1)
    {
      ASSERTFAIL("Go talk to Steve and tell him to implement this code real quick\n");
    }
    ASSERTFAIL("Not finished");
    if(stream.reading())
	adjust();
    stream.end_class();
}    

bool GeomGrid::saveobj(ostream&, const string& format, GeomSave*)
{
    NOT_FINISHED("GeomGrid::saveobj");
    return false;
}

void GeomGrid::compute_normals()
{
    hf_float_s6(&data[0], nu, nv);
}

} // End namespace GeomSpace
} // End namespace Core

#endif

#include <Core/Geom/GeomGrid.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomSave.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::ostream;
#include <stdio.h>

namespace SCIRun {


Persistent* make_GeomGrid()
{
    return scinew GeomGrid(0,0,Point(0,0,0), Vector(1,0,0), Vector(0,0,1));
}

PersistentTypeID GeomGrid::type_id("GeomGrid", "GeomObj", make_GeomGrid);

GeomGrid::GeomGrid(int nu, int nv, const Point& corner,
		   const Vector& u, const Vector& v, int image)
: verts(nu, nv), corner(corner), u(u), v(v), image(image)
{
    have_matls=0;
    have_normals=0;
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
}

void GeomGrid::set(int i, int j, double v)
{
    verts(i, j)=v;
}

double GeomGrid::get(int i, int j)
{
  double v = verts(i, j);
  //printf("Get: %d %d = %lf\n",i,j, v);
  return v;
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
    if (!image) {
	Vector uu(u/(nu-1));
	Vector vv(v/(nv-1));
	Point rstart(corner);
	for(int i=0;i<nu;i++){
	    Point p(rstart);
	    for(int j=0;j<nv;j++){
		Point pp(p+w*verts(i, j));
		bb.extend(pp);
		p+=vv;
	    }
	    rstart+=uu;
	}
    } else {
	bb.extend(corner-Vector(0.001,0.001,0.001));
	bb.extend(corner+u+v+Vector(0.001,0.001,0.001));
    }
}

GeomObj* GeomGrid::clone()
{
    return scinew GeomGrid(*this);
}

#define GEOMGRID_VERSION 1

void GeomGrid::io(Piostream& stream)
{

    stream.begin_class("GeomGrid", GEOMGRID_VERSION);
    GeomObj::io(stream);
    Pio(stream, verts);
    Pio(stream, have_matls);
    Pio(stream, matls);
    Pio(stream, have_normals);
    Pio(stream, normals);
    Pio(stream, corner);
    Pio(stream, u);
    Pio(stream, v);
    if(stream.reading())
	adjust();
    stream.end_class();
}    

bool GeomGrid::saveobj(ostream& out, const string& format,
		       GeomSave* saveinfo)
{
	int i,j;
    if(format == "vrml" || format == "iv"){    
	int nu=verts.dim1();
	int nv=verts.dim2();
	Vector uu(u/(nu-1));
	Vector vv(v/(nv-1));
	saveinfo->start_sep(out);
	saveinfo->start_node(out, "Coordinate3");
	saveinfo->indent(out);
	out << "point [";
	Point rstart(corner);
	for(i=0;i<nu;i++){
	    Point p1(rstart);
	    for(j=0;j<nv;j++){
		Point pp1(p1+w*verts(i, j));
		if(i>0 || j>0)
		    out << ", ";
		out << '\n';
		saveinfo->indent(out);
		out << pp1.x() << " " << pp1.y() << " " << pp1.z();
		p1+=vv;
	    }
	    rstart+=uu;
	}
	out << '\n';
	saveinfo->indent(out);
	out << " ]\n";
	saveinfo->end_node(out);
	saveinfo->start_node(out, "NormalBinding");
	saveinfo->indent(out);
	if(have_normals){
	    out << "value PER_VERTEX_INDEXED\n";
	} else {
	    out << "value OVERALL\n";
	}
	saveinfo->end_node(out);
	saveinfo->start_node(out, "Normal");
	saveinfo->indent(out);
	if(have_normals){
	    out << "vector [\n";
	    for(i=0;i<nu;i++){
		for(j=0;j<nv;j++){
		    if(i>0 || j>0){
			out << ',';
		    }	
		    out << '\n';
		    saveinfo->indent(out);
		    //Vector& normal(normals(i, j));  // Visual C++ hates this reference constructor syntax
			Vector& normal = normals(i,j);
		    out << normal.x() << ' ' << normal.y() << ' ' << normal.z();
		}
	    }
	    saveinfo->indent(out);
	    out << " ]\n";
	} else {
	    out << "vector " << w.x() << ' ' << w.y() << ' ' << w.z() << '\n';
	}
	saveinfo->end_node(out);
	saveinfo->start_node(out, "MaterialBinding");
	saveinfo->indent(out);
	if(have_matls){
	    out << "value PER_VERTEX_INDEXED\n";
	} else {
	    out << "value OVERALL\n";
	}
	saveinfo->end_node(out);	
	if(have_matls){
	    saveinfo->start_node(out, "Material");
	    saveinfo->indent(out);
	    out << "diffuseColor [\n";
	    for(i=0;i<nu-1;i++){
		for(j=0;j<nv-1;j++){
		    if(i>0 || j>0){
			out << ',';
		    }	
		    out << '\n';
		    saveinfo->indent(out);
		    float c[4];
		    matls(i,j)->diffuse.get_color(c);
		    out << c[0] << ' ' << c[1] << ' ' << c[2] << ' ' << c[3];
		}
	    }
	    saveinfo->indent(out);
	    out << " ]\n";
	    saveinfo->end_node(out);
	}

	saveinfo->start_node(out, "IndexedFaceSet");
	saveinfo->indent(out);
	out << "coordIndex [";
	for(i=0;i<nu-1;i++){
	    for(j=0;j<nv-1;j++){
		int i1=i*nv+j;
		int i2=i1+1;
		int i3=i1+nv+1;
		int i4=i1+nv;
		if(i>0 || j>0){
		    out << ',';
		}	
		out << '\n';
		saveinfo->indent(out);
		out << i1 << ", " << i2 << ", " << i3 << ", " << i4 << ", -1";
	    }
	}
	out << '\n';
	saveinfo->indent(out);
	out << " ]\n";
	if(have_matls){
	    saveinfo->indent(out);
	    out << "materialIndex [";
	    for(i=0;i<nu-1;i++){
		for(j=0;j<nv-1;j++){
		    int i1=i*nv+j;
		    int i2=i1+1;
		    int i3=i1+nv+1;
		    int i4=i1+nv;
		    if(i>0 || j>0){
			out << ',';
		    }
		    out << '\n';
		    saveinfo->indent(out);
		    out << i1 << ", " << i2 << ", " << i3 << ", " << i4 << ", -1";
		}
	    }
	    out << '\n';
	    saveinfo->indent(out);
	    out << "]\n";
	}
	if(have_normals){
	    saveinfo->indent(out);
	    out << "normalIndex [";
	    for(i=0;i<nu-1;i++){
		for(j=0;j<nv-1;j++){
		    int i1=i*nv+j;
		    int i2=i1+1;
		    int i3=i1+nv+1;
		    int i4=i1+nv;
		    if(i>0 || j>0){
			out << ',';
		    }
		    out << '\n';
		    saveinfo->indent(out);
		    out << i1 << ", " << i2 << ", " << i3 << ", " << i4 << ", -1";
		}
	    }
	    out << '\n';
	    saveinfo->indent(out);
	    out << "]\n";
	}
	saveinfo->end_node(out);
	saveinfo->end_sep(out);
	return true;
    } else {
	NOT_FINISHED("GeomGrid::saveobj");
	return false;
    }
}

} // End namespace SCIRun

