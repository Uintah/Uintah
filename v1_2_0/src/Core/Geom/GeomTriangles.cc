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
 *  GeomTriangles.cc: Triangle Strip object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Core/Geom/GeomTriangles.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomSave.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;
#include <stdio.h>

namespace SCIRun {

Persistent* make_GeomTriangles()
{
    return scinew GeomTriangles;
}

PersistentTypeID GeomTriangles::type_id("GeomTriangles", "GeomObj", make_GeomTriangles);

Persistent* make_GeomTrianglesP()
{
    return scinew GeomTrianglesP;
}

PersistentTypeID GeomTrianglesP::type_id("GeomTrianglesP", "GeomObj", make_GeomTrianglesP);

Persistent* make_GeomTrianglesPC()
{
    return scinew GeomTrianglesPC;
}

PersistentTypeID GeomTrianglesPC::type_id("GeomTrianglesPC", "GeomTrianglesP", make_GeomTrianglesPC);

Persistent* make_GeomTrianglesVP()
{
    return scinew GeomTrianglesVP;
}

PersistentTypeID GeomTrianglesVP::type_id("GeomTrianglesVP", "GeomObj", make_GeomTrianglesVP);

Persistent* make_GeomTrianglesVPC()
{
    return scinew GeomTrianglesVPC;
}

PersistentTypeID GeomTrianglesVPC::type_id("GeomTrianglesVPC", "GeomTrianglesVP", make_GeomTrianglesVPC);

Persistent* make_GeomTrianglesPT1d()
{
    return scinew GeomTrianglesPT1d;
}

PersistentTypeID GeomTrianglesPT1d::type_id("GeomTrianglesPT1d", "GeomObj", make_GeomTrianglesPT1d);

Persistent* make_GeomTranspTrianglesP()
{
    return scinew GeomTranspTrianglesP;
}

PersistentTypeID GeomTranspTrianglesP::type_id("GeomTranspTrianglesP", "GeomObj", make_GeomTranspTrianglesP);

GeomTriangles::GeomTriangles()
{
}

GeomTriangles::GeomTriangles(const GeomTriangles& copy)
: GeomVertexPrim(copy)
{
}

GeomTriangles::~GeomTriangles() {
}

void GeomTriangles::add(const Point& p1, const Point& p2, const Point& p3) {
    Vector n(Cross(p2-p1, p3-p1));
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", " << p2 << ", " << p3 << ")" << endl;
	return;
    }
#endif
    normals.add(n);
    GeomVertexPrim::add(p1);
    GeomVertexPrim::add(p2);
    GeomVertexPrim::add(p3);
}

int GeomTriangles::size(void)
{
    return verts.size();
}

void GeomTriangles::add(const Point& p1, const Vector& v1,
			const Point& p2, const Vector& v2,
			const Point& p3, const Vector& v3) {
    Vector n(Cross(p2-p1, p3-p1));
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", v1, " << p2 << ", v2, " << p3 << ", v3)" << endl;
	return;
    }
#endif
    normals.add(n);
    GeomVertexPrim::add(p1, v1);
    GeomVertexPrim::add(p2, v2);
    GeomVertexPrim::add(p3, v3);
}

void GeomTriangles::add(const Point& p1, const MaterialHandle& m1,
			const Point& p2, const MaterialHandle& m2,
			const Point& p3, const MaterialHandle& m3) {
    Vector n(Cross(p2-p1, p3-p1));
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", m1, " << p2 << ", m2, " << p3 << ", m3)" << endl;
	return;
    }
#endif
    normals.add(n);
    GeomVertexPrim::add(p1, m1);
    GeomVertexPrim::add(p2, m2);
    GeomVertexPrim::add(p3, m3);
}

void GeomTriangles::add(const Point& p1, const Color& c1,
			const Point& p2, const Color& c2,
			const Point& p3, const Color& c3) {
    Vector n(Cross(p2-p1, p3-p1));
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", c1, " << p2 << ", c2, " << p3 << ", c3)" << endl;
	return;
    }
#endif
    normals.add(n);
    GeomVertexPrim::add(p1, c1);
    GeomVertexPrim::add(p2, c2);
    GeomVertexPrim::add(p3, c3);
}

void GeomTriangles::add(const Point& p1, const Vector& v1, 
			const MaterialHandle& m1, const Point& p2, 
			const Vector& v2, const MaterialHandle& m2,
			const Point& p3, const Vector& v3, 
			const MaterialHandle& m3) {
    Vector n(Cross(p2-p1, p3-p1));
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", v1, m1, " << p2 << ", v2, m2, " << p3 << ", v3, m3)" << endl;
	return;
    }
#endif
    normals.add(n);
    GeomVertexPrim::add(p1, v1, m1);
    GeomVertexPrim::add(p2, v2, m2);
    GeomVertexPrim::add(p3, v3, m3);
}

void GeomTriangles::add(GeomVertex* v1, GeomVertex* v2, GeomVertex* v3) {
    Vector n(Cross(v3->p - v1->p, v2->p - v1->p));
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle in GeomTriangles::add(v1->" << v1->p << ", v2->" << v2->p << ", v3->" << v3->p << ")" << endl;
//	cerr << "Degenerate triangle!!!\n" << endl;
	return;
    }
#endif
    normals.add(n);
    GeomVertexPrim::add(v1);
    GeomVertexPrim::add(v2);
    GeomVertexPrim::add(v3);
}

GeomObj* GeomTriangles::clone()
{
    return scinew GeomTriangles(*this);
}

#define GEOMTRIANGLES_VERSION 1

void GeomTriangles::io(Piostream& stream)
{

    stream.begin_class("GeomTriangles", GEOMTRIANGLES_VERSION);
    GeomVertexPrim::io(stream);
    Pio(stream, normals);
    stream.end_class();
}

bool GeomTriangles::saveobj(ostream&, const string&, GeomSave*)
{
    NOT_FINISHED("GeomTriangles::saveobj");
    return false;
}

GeomTrianglesPT1d::GeomTrianglesPT1d()
:GeomTrianglesP(),cmap(0)
{
}

GeomTrianglesPT1d::~GeomTrianglesPT1d()
{
}

int GeomTrianglesPT1d::add(const Point& p1,const Point& p2,const Point& p3,
			   const float& f1,const float& f2,const float& f3)
{
  if (GeomTrianglesP::add(p1,p2,p3)) {
    scalars.add(f1);
    scalars.add(f2);
    scalars.add(f3);
    return 1;
  }
  return 0;
}


#define GeomTrianglesPT1d_VERSION 1

void GeomTrianglesPT1d::io(Piostream& stream)
{

    stream.begin_class("GeomTrianglesPT1d", GeomTrianglesPT1d_VERSION);
    GeomTrianglesP::io(stream);
    Pio(stream, scalars); // just save scalar values
    stream.end_class();
}

bool GeomTrianglesPT1d::saveobj(ostream& out, const string& format,
				GeomSave* saveinfo)
{
#if 0
    static int cnt=0;
    if(++cnt > 2)
	return true;
#endif
    if(format == "vrml" || format == "iv"){
	saveinfo->start_sep(out);
	saveinfo->start_node(out, "Coordinate3");
	saveinfo->indent(out);
	out << "point [";
	int np=size();
	int idx=0;
	int i;
	for(i=0;i<np*3;i++){
	    if(i>0)
		out << ", ";
	    out << '\n';
	    saveinfo->indent(out);
	    out << " " << points[idx] << " " << points[idx+1] << " " << points[idx+2];
	    idx+=3;
	}
	saveinfo->indent(out);
	out << " ]\n";
	saveinfo->end_node(out);
	saveinfo->start_node(out, "NormalBinding");
	saveinfo->indent(out);
	out << "value PER_FACE\n";
	saveinfo->end_node(out);
	saveinfo->start_node(out, "Normal");
	saveinfo->indent(out);
	out << "vector [";
	idx=0;
	for(i=0;i<np;i++){
	    if(i>0)
		out << ", ";
	    out << '\n';
	    saveinfo->indent(out);
	    out << " " << normals[idx] << " " << normals[idx+1] << " " << normals[idx+2];
	    idx+=3;
	}
	saveinfo->indent(out);
	out << " ]\n";
	saveinfo->end_node(out);
	saveinfo->start_node(out, "MaterialBinding");
	saveinfo->indent(out);
#if 0
	out << "value PER_FACE_INDEXED\n";
#else
	out << "value OVERALL\n";
#endif
	saveinfo->end_node(out);	
#if 0
	saveinfo->start_node(out, "Texture2");
	saveinfo->indent(out);
	out << "image  1 256 4 ";
	if(cmap){
	    unsigned char* p=cmap;
	    for(i=0;i<256;i++){
		char buf[20];
		out << '\n';
		saveinfo->indent(out);
		sprintf(buf, "0x%02x%02x%02x%02x ", p[0], p[1], p[2], 255-p[3]);
		out << buf;
		p+=4;
	    }
	} else {
	    cerr << "colormap not saved, making up a lame one\n";
	    for(i=0;i<256;i++){
		char buf[20];
		unsigned char r,g,b,a;
		r=i;
		b=255-r;
		g=0;
		a=255;
		out << '\n';
		saveinfo->indent(out);
		sprintf(buf, "0x%02x%02x%02x%02x ", r, g, b, a);
		out << buf;
	    }
	}
	saveinfo->end_node(out);
#endif
	saveinfo->start_node(out, "Material");
	saveinfo->indent(out);
	out << "ambientColor 0.2 0.2 0.2\n";
#if 0
	saveinfo->indent(out);
	out << "diffuseColor 0.8 0.8 0.8\n";
#else
#if 1
	saveinfo->indent(out);
	static int cc=0;
	cc++;
	cerr << "cc=" << cc << '\n';
	if(cc<=11)
	    out << "diffuseColor 0.8 0.0 0.0\n";
	else
	    out << "diffuseColor 0.0 0.8 0.0\n";
#else
	saveinfo->indent(out);
	out << "diffuseColor [\n";
	if(cmap){
	    unsigned char* p=cmap;
	    for(i=0;i<256;i++){
		if(i>0)
		    out << ',';
		out << '\n';
		saveinfo->indent(out);
		out << p[0]/255. << ' ' << p[1]/255. << ' ' << p[2]/255.;
		p+=4;
	    }
	} else {
	    cerr << "colormap not saved, making up a lame one\n";
	    for(i=0;i<256;i++){
		if(i>0)
		    out << ',';
		out << '\n';
		saveinfo->indent(out);
		out << i/255. << ' ' << 0. << ' ' << (255-i)/255.;
	    }
	}
	out << '\n';
	saveinfo->indent(out);
	out << "]\n";
#endif
#endif	
	saveinfo->indent(out);
	out << "specularColor 0.6 0.6 0.6\n";
	saveinfo->end_node(out);
#if 0
	saveinfo->start_node(out, "TextureCoordinate2");
	saveinfo->indent(out);
	out << "point [";
	for(i=0;i<np;i++){
	    if(i>0)
		out << ", ";
	    out << '\n';
	    saveinfo->indent(out);
	    out << " 0 " << scalars[i];
	}
	saveinfo->indent(out);
	out << " ]\n";
	saveinfo->end_node(out);
#endif
	saveinfo->start_node(out, "IndexedFaceSet");
	saveinfo->indent(out);
#if 0
	out << "textureCoordIndex [";
	for(i=0;i<np;i++){
	    if(i>0)
		out << ",";
	    out << '\n';
	    saveinfo->indent(out);
	    out << " " << i << ", " << i << ", " << i << ", " << -1;
	}
	out << " ]\n";
#else
	out << "materialIndex [";
	for(i=0;i<np;i++){
	    if(i>0)
		out << ",";
	    if(i%10==0){
		out << '\n';
		saveinfo->indent(out);
	    }
	    int idx=(int)(scalars[i]*255.);
	    if(idx<0)
		idx=0;
	    else if(idx>255)
		idx=255;
	    out << idx;
	}
	out << " ]\n";
#endif
	out << "coordIndex [";
	for(i=0;i<3*np;i++){
	    if(i>0){
		out << ",";
		if(i%3==0){
		    out << '\n';
		    saveinfo->indent(out);
		}
	    }
	    out << " " << i;
	    if(i%3==2) {
		out << ", -1";
	    }
	}
	saveinfo->indent(out);
	out << " ]\n";
	saveinfo->indent(out);
	saveinfo->end_node(out);
	saveinfo->start_node(out, "Texture2");
	saveinfo->indent(out);
	out << "filename \"\"\n";
	saveinfo->end_node(out);
	saveinfo->end_sep(out);
	return true;
    } else {
	NOT_FINISHED("GeomTrianglesPT1d::saveobj");
	return false;
    }
}

GeomTranspTrianglesP::GeomTranspTrianglesP()
  : alpha(0.2), sorted(0), list_pos(0)
{
}

GeomTranspTrianglesP::GeomTranspTrianglesP(double aval)
  : alpha(aval), sorted(0), list_pos(0)
{
}

GeomTranspTrianglesP::~GeomTranspTrianglesP()
{
  // nothing to worry about...
}

int GeomTranspTrianglesP::vadd(const Point& p1, 
			       const Point& p2, const Point& p3)
{
  if (add(p1,p2,p3)) {
    Vector center = (p1.vector()+p2.vector()+p3.vector())*(1.0/3.0);
    xc.add(center.x());
    yc.add(center.y());
    zc.add(center.z());
    return 1;
  } 
  return 0;
}


struct SortHelper {
  static float* ctr_array;
  int                  id; // id for this guy...
};

float * SortHelper::ctr_array = 0;

int SimpComp(const void* e1, const void* e2)
{
  SortHelper *a = (SortHelper*)e1;
  SortHelper *b = (SortHelper*)e2;

  if (SortHelper::ctr_array[a->id] >
      SortHelper::ctr_array[b->id])
    return 1;
  if (SortHelper::ctr_array[a->id] <
      SortHelper::ctr_array[b->id])
    return -1;

  return 0; // they are equal...  
}

void GeomTranspTrianglesP::SortPolys()
{
  cerr << "Starting sort...\n";
  
  SortHelper::ctr_array = &xc[0];

  xlist.setsize(size());
  ylist.setsize(size());
  zlist.setsize(size());

  Array1<SortHelper> help;

  help.resize(size());

  int i;
  for(i=0;i<size();i++) {
    help[i].id = i;
  }

  qsort(&help[0],help.size(),sizeof(SortHelper),SimpComp);

  // now dump them back...

  for(i=0;i<size();i++) {
    xlist[i] = help[i].id;
    help[i].id = i; // reset for next pass...
  }

  SortHelper::ctr_array = &yc[0];

  for(i=0;i<size();i++) {
    help[i].id = i;
  }

  qsort(&help[0],help.size(),sizeof(SortHelper),SimpComp);

  // now dump them back...

  for(i=0;i<size();i++) {
    ylist[i] = help[i].id;
    help[i].id = i; // reset for next pass...
  }

  SortHelper::ctr_array = &zc[0];

  for(i=0;i<size();i++) {
    help[i].id = i;
  }

  qsort(&help[0],help.size(),sizeof(SortHelper),SimpComp);

  // now dump them back...

  for(i=0;i<size();i++) {
    zlist[i] = help[i].id;
  }

  cerr << "Done sorting!\n";
  
}

// grows points, normals and centers...

void GeomTranspTrianglesP::MergeStuff(GeomTranspTrianglesP* other)
{
  points.resize(points.size() + other->points.size());
  normals.resize(normals.size() + other->normals.size());

  xc.resize(xc.size() + other->xc.size());
  yc.resize(yc.size() + other->yc.size());
  zc.resize(zc.size() + other->zc.size());
  
  int start = points.size()-other->points.size();
  int i;
  for(i=0;i<other->points.size();i++) {
    points[i+start] = other->points[i];
  }
  start = normals.size() - other->normals.size();
  for(i=0;i<other->normals.size();i++) {
    normals[i+start] = other->normals[i];
  }

  start = xc.size() - other->xc.size();
  for(i=0;i<other->xc.size();i++) {
    xc[start +i] = other->xc[i];
    yc[start +i] = other->yc[i];
    zc[start +i] = other->zc[i];
  }
  other->points.resize(0);
  other->normals.resize(0);
  other->xc.resize(0);
  other->yc.resize(0);
  other->zc.resize(0);
}

#define GeomTranspTrianglesP_VERSION 1

void GeomTranspTrianglesP::io(Piostream& stream)
{

    stream.begin_class("GeomTranspTrianglesP", GeomTranspTrianglesP_VERSION);
    GeomTrianglesP::io(stream);
    Pio(stream, alpha); // just save transparency value...
    stream.end_class();
}

GeomTrianglesP::GeomTrianglesP()
:has_color(0)
{
    // don't really need to do anythin...
}

GeomTrianglesP::~GeomTrianglesP()
{

}

void GeomTrianglesP::get_triangles( Array1<float> &v)
{
  int end = v.size();
  v.grow (points.size());
  for (int i=0; i<points.size(); i++)
    v[end+i] = points[i];
}

int GeomTrianglesP::size(void)
{
    return points.size()/9;
}

void GeomTrianglesP::reserve_clear(int n)
{
    points.setsize(n*9);
    normals.setsize(n*3);

    points.remove_all();
    normals.remove_all();
}

int GeomTrianglesP::add(const Point& p1, const Point& p2, const Point& p3)
{
    Vector n(Cross(p2-p1, p3-p1));
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
        n.normalize();
    }   	
    else {
//	cerr << "degenerate triangle!!!\n" << endl;
	return 0;
    }
#endif

    int idx=normals.size();
    normals.grow(3);
    normals[idx]=n.x();
    normals[idx+1]=n.y();
    normals[idx+2]=n.z();


    idx=points.size();
    points.grow(9);
    points[idx]=p1.x();
    points[idx+1]=p1.y();
    points[idx+2]=p1.z();
    points[idx+3]=p2.x();
    points[idx+4]=p2.y();
    points[idx+5]=p2.z();
    points[idx+6]=p3.x();
    points[idx+7]=p3.y();
    points[idx+8]=p3.z();
    return 1;
}

// below is just a virtual function...

int GeomTrianglesP::vadd(const Point& p1, const Point& p2, const Point& p3)
{
    Vector n(Cross(p2-p1, p3-p1));
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
        n.normalize();
    }   	
    else {
//	cerr << "degenerate triangle!!!\n" << endl;
	return 0;
    }
#endif

    int idx=normals.size();
    normals.grow(3);
    normals[idx]=n.x();
    normals[idx+1]=n.y();
    normals[idx+2]=n.z();


    idx=points.size();
    points.grow(9);
    points[idx]=p1.x();
    points[idx+1]=p1.y();
    points[idx+2]=p1.z();
    points[idx+3]=p2.x();
    points[idx+4]=p2.y();
    points[idx+5]=p2.z();
    points[idx+6]=p3.x();
    points[idx+7]=p3.y();
    points[idx+8]=p3.z();
    return 1;
}

GeomObj* GeomTrianglesP::clone()
{
    return new GeomTrianglesP(*this);
}

void GeomTrianglesP::get_bounds(BBox& box)
{
    for(int i=0;i<points.size();i+=3)
	box.extend(Point(points[i],points[i+1],points[i+2]));
}

#define GEOMTRIANGLESP_VERSION 1

void GeomTrianglesP::io(Piostream& stream)
{

    stream.begin_class("GeomTrianglesP", GEOMTRIANGLESP_VERSION);
    GeomObj::io(stream);
    Pio(stream, points);
    Pio(stream, normals);
    stream.end_class();
}

bool GeomTrianglesP::saveobj(ostream&, const string&, GeomSave*)
{
    NOT_FINISHED("GeomTrianglesP::saveobj");
    return false;
}

GeomTrianglesPC::GeomTrianglesPC()
{
    // don't really need to do anythin...
}

GeomTrianglesPC::~GeomTrianglesPC()
{

}

int GeomTrianglesPC::add(const Point& p1, const Color& c1,
			const Point& p2, const Color& c2,
			const Point& p3, const Color& c3)
{
    if (GeomTrianglesP::add(p1,p2,p3)) {
	colors.add(c1.r());
	colors.add(c1.g());
	colors.add(c1.b());

	colors.add(c2.r());
	colors.add(c2.g());
	colors.add(c2.b());

	colors.add(c3.r());
	colors.add(c3.g());
	colors.add(c3.b());
	return 1;
    }

    return 0;
}

#define GEOMTRIANGLESPC_VERSION 1

void GeomTrianglesPC::io(Piostream& stream)
{

    stream.begin_class("GeomTrianglesPC", GEOMTRIANGLESPC_VERSION);
    GeomTrianglesP::io(stream);
    Pio(stream, colors);
    stream.end_class();
}


bool GeomTrianglesPC::saveobj(ostream& out, const string& format,
			      GeomSave* saveinfo)
{
    if(format == "vrml"){
	saveinfo->start_sep(out);
	saveinfo->start_node(out, "Coordinate3");
	saveinfo->indent(out);
	out << "point [";
	int np=points.size()/3;
	int i;
	for(i=0;i<np;i++){
	    if(i>0)
		out << ", ";
	    int idx=i*3;
	    out << " " << points[idx] << " " << points[idx+1] << " " << points[idx+2];
	}
	out << " ]\n";
	saveinfo->end_node(out);
	saveinfo->start_node(out, "NormalBinding");
	saveinfo->indent(out);
	out << "value PER_VERTEX\n";
	saveinfo->end_node(out);
	saveinfo->start_node(out, "Normal");
	saveinfo->indent(out);
	out << "vector [";
	for(i=0;i<np;i+=3){
	    if(i>0)
		out << ", ";
	    int idx=i;
	    out << " " << normals[idx] << " " << normals[idx+1] << " " << normals[idx+2];
	}
	out << " ]\n";
	saveinfo->end_node(out);
	saveinfo->start_node(out, "MaterialBinding");
	saveinfo->indent(out);
	out << "value PER_VERTEX\n";
	saveinfo->end_node(out);
	saveinfo->start_node(out, "Material");
	saveinfo->indent(out);
	out << "diffuseColor [";
	for(i=0;i<np;i++){
	    if(i>0)
		out << ", ";
	    int idx=i*3;
	    out << " " << colors[idx] << " " << colors[idx+1] << " " << colors[idx+2];
	}
	out << " ]\n";
	saveinfo->end_node(out);
	saveinfo->start_node(out, "IndexedFaceSet");
	saveinfo->indent(out);
	out << "coordIndex [";
	for(i=0;i<np;i++){
	    if(i>0)
		out << ", ";
	    out << " " << i;
	    if(i%3==2)
		out << ", -1";
	}
	out << " ]\n";
	saveinfo->end_node(out);
	saveinfo->end_sep(out);
	return true;
    } else {
	NOT_FINISHED("GeomTrianglesPC::saveobj");
	return false;
    }
}

GeomTrianglesVP::GeomTrianglesVP()
{
    // don't really need to do anythin...
}

GeomTrianglesVP::~GeomTrianglesVP()
{

}

int GeomTrianglesVP::size(void)
{
    return points.size()/9;
}

void GeomTrianglesVP::reserve_clear(int n)
{
    int np = points.size()/9;
    int delta = n - np;

    points.remove_all();
    normals.remove_all();

    if (delta > 0) {
	points.grow(delta);
	normals.grow(delta);
    }
	
}

int GeomTrianglesVP::add(const Point& p1, const Vector &v1,
			 const Point& p2, const Vector &v2,	
			 const Point& p3, const Vector &v3)
{
    int idx=normals.size();
    normals.grow(9);
    normals[idx]=v1.x();
    normals[idx+1]=v1.y();
    normals[idx+2]=v1.z();
    normals[idx+3]=v2.x();
    normals[idx+4]=v2.y();
    normals[idx+5]=v2.z();
    normals[idx+6]=v3.x();
    normals[idx+7]=v3.y();
    normals[idx+8]=v3.z();

    idx=points.size();
    points.grow(9);
    points[idx]=p1.x();
    points[idx+1]=p1.y();
    points[idx+2]=p1.z();
    points[idx+3]=p2.x();
    points[idx+4]=p2.y();
    points[idx+5]=p2.z();
    points[idx+6]=p3.x();
    points[idx+7]=p3.y();
    points[idx+8]=p3.z();
    return 1;
}

GeomObj* GeomTrianglesVP::clone()
{
    return new GeomTrianglesVP(*this);
}

void GeomTrianglesVP::get_bounds(BBox& box)
{
    for(int i=0;i<points.size();i+=3)
	box.extend(Point(points[i],points[i+1],points[i+2]));
}

#define GEOMTRIANGLESVP_VERSION 1

void GeomTrianglesVP::io(Piostream& stream)
{

    stream.begin_class("GeomTrianglesVP", GEOMTRIANGLESVP_VERSION);
    GeomObj::io(stream);
    Pio(stream, points);
    Pio(stream, normals);
    stream.end_class();
}

bool GeomTrianglesVP::saveobj(ostream&, 
			      const string& /*format*/,
			      GeomSave*)
{
    NOT_FINISHED("GeomTrianglesVP::saveobj");
    return false;
}

GeomTrianglesVPC::GeomTrianglesVPC()
{
    // don't really need to do anythin...
}

GeomTrianglesVPC::~GeomTrianglesVPC()
{

}

int GeomTrianglesVPC::add(const Point& p1, const Vector &v1, const Color& c1,
			  const Point& p2, const Vector &v2, const Color& c2,
			  const Point& p3, const Vector &v3, const Color& c3)
{
    if (GeomTrianglesVP::add(p1,v1,p2,v2,p3,v3)) {
	colors.add(c1.r());
	colors.add(c1.g());
	colors.add(c1.b());

	colors.add(c2.r());
	colors.add(c2.g());
	colors.add(c2.b());

	colors.add(c3.r());
	colors.add(c3.g());
	colors.add(c3.b());
	return 1;
    }

    return 0;
}

#define GEOMTRIANGLESVPC_VERSION 1

void GeomTrianglesVPC::io(Piostream& stream)
{

    stream.begin_class("GeomTrianglesVPC", GEOMTRIANGLESVPC_VERSION);
    GeomTrianglesVP::io(stream);
    Pio(stream, colors);
    stream.end_class();
}


bool GeomTrianglesVPC::saveobj(ostream& out, const string& format,
			      GeomSave* saveinfo)
{
    if(format == "vrml" || format == "iv"){
	saveinfo->start_sep(out);
	saveinfo->start_node(out, "Coordinate3");
	saveinfo->indent(out);
	out << "point [";
	int np=points.size()/3;
	int i;
	for(i=0;i<np;i++){
	    if(i>0)
		out << ", ";
	    int idx=i*3;
	    out << " " << points[idx] << " " << points[idx+1] << " " << points[idx+2];
	}
	out << " ]\n";
	saveinfo->end_node(out);
	saveinfo->start_node(out, "NormalBinding");
	saveinfo->indent(out);
	out << "value PER_VERTEX\n";
	saveinfo->end_node(out);
	saveinfo->start_node(out, "Normal");
	saveinfo->indent(out);
	out << "vector [";
	for(i=0;i<np;i+=3){
	    if(i>0)
		out << ", ";
	    int idx=i;
	    out << " " << normals[idx] << " " << normals[idx+1] << " " << normals[idx+2];
	}
	out << " ]\n";
	saveinfo->end_node(out);
	saveinfo->start_node(out, "MaterialBinding");
	saveinfo->indent(out);
	out << "value PER_VERTEX\n";
	saveinfo->end_node(out);
	saveinfo->start_node(out, "Material");
	saveinfo->indent(out);
	out << "diffuseColor [";
	for(i=0;i<np;i++){
	    if(i>0)
		out << ", ";
	    int idx=i*3;
	    out << " " << colors[idx] << " " << colors[idx+1] << " " << colors[idx+2];
	}
	out << " ]\n";
	saveinfo->end_node(out);
	saveinfo->start_node(out, "IndexedFaceSet");
	saveinfo->indent(out);
	out << "coordIndex [";
	for(i=0;i<np;i++){
	    if(i>0)
		out << ", ";
	    out << " " << i;
	    if(i%3==2)
		out << ", -1";
	}
	out << " ]\n";
	saveinfo->end_node(out);
	saveinfo->end_sep(out);
	return true;
    } else {
	NOT_FINISHED("GeomTrianglesVPC::saveobj");
	return false;
    }
}

} // End namespace SCIRun

