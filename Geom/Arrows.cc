
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
#include <Classlib/String.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>
#include <Malloc/Allocator.h>

Persistent* make_GeomArrows()
{
    return scinew GeomArrows(0,0);
}

PersistentTypeID GeomArrows::type_id("GeomArrows", "GeomObj", make_GeomArrows);

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

#define GEOMARROWS_VERSION 1

void GeomArrows::io(Piostream& stream)
{
    stream.begin_class("GeomArrows", GEOMARROWS_VERSION);
    GeomObj::io(stream);
    Pio(stream, headwidth);
    Pio(stream, headlength);
    Pio(stream, shaft_matls);
    Pio(stream, back_matls);
    Pio(stream, head_matls);
    Pio(stream, positions);
    Pio(stream, directions);
    Pio(stream, v1);
    Pio(stream, v2);
    stream.end_class();
}

bool GeomArrows::saveobj(ostream&, const clString& format, GeomSave*)
{
#if 0
  int n=positions.size();

  // Draw shafts - they are the same for all draw types....
  double shaft_scale=headlength;

  if(shaft_matls.size() == 1){
    // Material is same across all arrows.
    // Output the material.
    // You need to make a generic material outputter.
#if 0
	matl->ambient.get_color(color);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color);
	matl->diffuse.get_color(color);
	glColor4fv(color);
	matl->specular.get_color(color);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, color);
	matl->emission.get_color(color);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, matl->shininess);
	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, color);
#endif

    // Output each line.
    for(int i=0;i<n;i++){
      Point from(positions[i]);
      Point to(from+directions[i]*shaft_scale);
      // From -> to.
    }
  } else {
    // Output each line.
    for(int i=0;i<n;i++){
      // Output the material.
      Point from(positions[i]);
      Point to(from+directions[i]*shaft_scale);
      // From -> to.
    }
  }

  // Draw the back.
  if(back_matls.size() == 1){
    // Color is same on each.
    glBegin(GL_QUADS);
    for(int i=0;i<n;i++){
      Point from(positions[i]+directions[i]*headlength);
      glNormal3d(directions[i].x(), directions[i].y(), directions[i].z());
      Point to(from+directions[i]);
      Point p1(from+v1[i]);
      glVertex3d(p1.x(), p1.y(), p1.z());
      Point p2(from+v2[i]);
      glVertex3d(p2.x(), p2.y(), p2.z());
      Point p3(from-v1[i]);
      glVertex3d(p3.x(), p3.y(), p3.z());
      Point p4(from-v2[i]);
      glVertex3d(p4.x(), p4.y(), p4.z());
    }
  } else {
    glBegin(GL_QUADS);
    for(int i=0;i<n;i++){
      di->set_matl(back_matls[i].get_rep());
      glNormal3d(directions[i].x(), directions[i].y(), directions[i].z());
      Point from(positions[i]+directions[i]*headlength);
      Point to(from+directions[i]);
      Point p1(from+v1[i]);
      glVertex3d(p1.x(), p1.y(), p1.z());
      Point p2(from+v2[i]);
      glVertex3d(p2.x(), p2.y(), p2.z());
      Point p3(from-v1[i]);
      glVertex3d(p3.x(), p3.y(), p3.z());
      Point p4(from-v2[i]);
      glVertex3d(p4.x(), p4.y(), p4.z());
    }
  }

  // Draw the head.
  if(head_matls.size() == 1){
    double w=headwidth;
    double h=1.0-headlength;
    double w2h2=w*w/h;
    for(int i=0;i<n;i++){
      glBegin(GL_TRIANGLES);
      Vector dn(directions[i]*w2h2);
      Vector n(dn+v1[i]+v2[i]);
      glNormal3d(n.x(), n.y(), n.z());

      Point top(positions[i]+directions[i]);
      Point from=top-directions[i]*h;
      Point to(from+directions[i]);
      Point p1(from+v1[i]);
      Point p2(from+v2[i]);
      glVertex3d(top.x(), top.y(), top.z());
      glVertex3d(p1.x(), p1.y(), p1.z());
      glVertex3d(p2.x(), p2.y(), p2.z()); // 1st tri
      n=dn-v1[i]+v2[i];
      glNormal3d(n.x(), n.y(), n.z());
      Point p3(from-v1[i]);
      glVertex3d(top.x(), top.y(), top.z());
      glVertex3d(p2.x(), p2.y(), p2.z());
      glVertex3d(p3.x(), p3.y(), p3.z()); // 2nd tri
      n=dn-v1[i]-v2[i];
      glNormal3d(n.x(), n.y(), n.z());
      Point p4(from-v2[i]);
      glVertex3d(top.x(), top.y(), top.z());
      glVertex3d(p3.x(), p3.y(), p3.z());
      glVertex3d(p4.x(), p4.y(), p4.z()); // 3rd tri
      n=dn+v1[i]-v2[i];
      glNormal3d(n.x(), n.y(), n.z());
      glVertex3d(top.x(), top.y(), top.z());
      glVertex3d(p4.x(), p4.y(), p4.z());
      glVertex3d(p1.x(), p1.y(), p1.z()); // 4th tri
    }
  } else {
    double w=headwidth;
    double h=1.0-headlength;
    double w2h2=w*w+h*h;
    for(int i=0;i<n;i++){
      glBegin(GL_TRIANGLES);
      Vector dn(directions[i]*w2h2);
      Vector n(dn+v1[i]+v2[i]);
      glNormal3d(n.x(), n.y(), n.z());
      di->set_matl(back_matls[i].get_rep());

      Point top(positions[i]+directions[i]);
      Point from=top-directions[i]*h;
      Point to(from+directions[i]);
      Point p1(from+v1[i]);
      Point p2(from+v2[i]);
      glVertex3d(top.x(), top.y(), top.z());
      glVertex3d(p1.x(), p1.y(), p1.z());
      glVertex3d(p2.x(), p2.y(), p2.z()); // 1st tri
      n=dn-v1[i]+v2[i];
      glNormal3d(n.x(), n.y(), n.z());
      Point p3(from-v1[i]);
      glVertex3d(top.x(), top.y(), top.z());
      glVertex3d(p2.x(), p2.y(), p2.z());
      glVertex3d(p3.x(), p3.y(), p3.z()); // 2nd tri
      n=dn-v1[i]-v2[i];
      glNormal3d(n.x(), n.y(), n.z());
      Point p4(from-v2[i]);
      glVertex3d(top.x(), top.y(), top.z());
      glVertex3d(p3.x(), p3.y(), p3.z());
      glVertex3d(p4.x(), p4.y(), p4.z()); // 3rd tri
      n=dn+v1[i]-v2[i];
      glNormal3d(n.x(), n.y(), n.z());
      glVertex3d(top.x(), top.y(), top.z());
      glVertex3d(p4.x(), p4.y(), p4.z());
      glVertex3d(p1.x(), p1.y(), p1.z()); // 4th tri
      glEnd();
    }
  }
#endif
  return true;
}


#ifdef __GNUG__

#include <Classlib/Array1.cc>
template class Array1<Point>;
template class Array1<Vector>;

template void Pio(Piostream&, Array1<Point>&);
template void Pio(Piostream&, Array1<Vector>&);

#endif

#ifdef __sgi
#if _MIPS_SZPTR == 64
#include <Classlib/Array1.cc>

static void _dummy1_(Piostream& p1, Array1<Vector>& p2)
{
    Pio(p1, p2);
}

static void _dummy2_(Piostream& p1, Array1<Point>& p2)
{
    Pio(p1, p2);
}

#endif
#endif

