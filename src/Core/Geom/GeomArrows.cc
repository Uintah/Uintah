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
 *  GeomArrows.cc: Arrows object
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Core/Geom/GeomArrows.h>

#include <Core/Geom/GeomSave.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Persistent/Persistent.h>
#include <iostream>
using std::ostream;

#include <stdio.h>

namespace SCIRun {

Persistent* make_GeomArrows()
{
    return scinew GeomArrows(0,0,0);
}

PersistentTypeID GeomArrows::type_id("GeomArrows", "GeomObj", make_GeomArrows);

GeomArrows::GeomArrows(double headwidth, double headlength, int cyl, double r,
		       int normhead)
  : headwidth(headwidth), headlength(headlength), rad(r), drawcylinders(cyl),
    normalize_headsize(normhead)
{
    shaft_matls.add(new Material(Color(0,0,0), Color(.6, .6, .6), Color(.6, .6, .6), 10));
    head_matls.add(new Material(Color(0,0,0), Color(1,0,0), Color(.6, .6, .6), 10));
    back_matls.add(new Material(Color(0,0,0), Color(.6, .6, .6), Color(.6, .6, .6), 10));
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
  if(dir.length2() > 0) {
    positions.add(pos);
    directions.add(dir);
    Vector vv1, vv2;
    dir.find_orthogonal(vv1, vv2);
    if (!normalize_headsize) {
      // use the length to scale the head
      double len = dir.length();
      v1.add(vv1*headwidth*len);
      v2.add(vv2*headwidth*len);
    } else {
      // don't scale the head by the length
      v1.add(vv1*headwidth);
      v2.add(vv2*headwidth);
    }
  }
}

void GeomArrows::get_bounds(BBox& bb)
{
    int n=positions.size();
    for(int i=0;i<n;i++){
	bb.extend(positions[i]);
	bb.extend(positions[i]+directions[i]);
    }
}

GeomObj* GeomArrows::clone()
{
    return scinew GeomArrows(*this);
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

bool GeomArrows::saveobj(ostream& out, const string&, GeomSave* saveinfo)
{
  int n=positions.size();

  //////////////////////////////////////////////////////////////////////
  // Draw the lines.
  if(shaft_matls.size() == 1){
    // Material is same across all arrows.
    // Output the material.

    saveinfo->start_attr(out);

    saveinfo->indent(out);
    out << "Color [ " << shaft_matls[0]->diffuse.r() << " "
	<< shaft_matls[0]->diffuse.g() << " "
	<< shaft_matls[0]->diffuse.b() << " ]\n";

    saveinfo->indent(out);
    out << "Surface \"plastic\" \"Ka\" 0.0 \"Kd\" 1.0 \"Ks\" 1.0 \"roughness\" "
	<< 1.0 / shaft_matls[0]->shininess << " \"specularcolor\" [ "
	<< shaft_matls[0]->specular.r() << " "
	<< shaft_matls[0]->specular.g() << " "
	<< shaft_matls[0]->specular.b() << " ]\n";

    // Have to output them as cylinders.
    for(int i=0;i<n;i++){
      saveinfo->start_trn(out);
      saveinfo->rib_orient(out, positions[i], directions[i]);
      saveinfo->indent(out);
      out << "Cylinder " << 0.002 << " 0 " << directions[i].length() * headlength
	  << " 360\n";
      saveinfo->end_trn(out);
    }

    saveinfo->end_attr(out);
  } else {
    saveinfo->start_attr(out);

    // Output each line.
    for(int i=0;i<n;i++){
      // Output the material.
      
      saveinfo->indent(out);
      out << "Color [ " << shaft_matls[i]->diffuse.r() << " "
	  << shaft_matls[i]->diffuse.g() << " "
	  << shaft_matls[i]->diffuse.b() << " ]\n";

      saveinfo->indent(out);
      out << "Surface \"plastic\" \"Ka\" 0.0 \"Kd\" 1.0 \"Ks\" 1.0 \"roughness\" "
	  << 1.0 / shaft_matls[i]->shininess << " \"specularcolor\" [ "
	  << shaft_matls[i]->specular.r() << " "
	  << shaft_matls[i]->specular.g() << " "
	  << shaft_matls[i]->specular.b() << " ]\n";
      
      saveinfo->start_trn(out);
      saveinfo->rib_orient(out, positions[i], directions[i]);
      saveinfo->indent(out);
      out << "Cylinder " << 0.002 << " 0 " << directions[i].length() * headlength
	  << " 360\n";
      saveinfo->end_trn(out);
      
    }
    saveinfo->end_attr(out);
  }

  //////////////////////////////////////////////////////////////////////
  // Draw the back.
  if(back_matls.size() == 1){
    // Color is same on each.

    saveinfo->start_attr(out);

    saveinfo->indent(out);
    out << "Color [ " << back_matls[0]->diffuse.r() << " "
	<< back_matls[0]->diffuse.g() << " "
	<< back_matls[0]->diffuse.b() << " ]\n";

    saveinfo->indent(out);
    out << "Surface \"plastic\" \"Ka\" 0.0 \"Kd\" 1.0 \"Ks\" 1.0 \"roughness\" "
	<< 1.0 / back_matls[0]->shininess << " \"specularcolor\" [ "
	<< back_matls[0]->specular.r() << " "
	<< back_matls[0]->specular.g() << " "
	<< back_matls[0]->specular.b() << " ]\n";

    for(int i=0;i<n;i++){
      Point from(positions[i]+directions[i]*headlength);

      Point p1(from+v1[i]);
      Point p2(from+v2[i]);
      Point p3(from-v1[i]);
      Point p4(from-v2[i]);
      
      saveinfo->indent(out);
      out << "Polygon \"P\" [ "
	  << p1.x() << " " << p1.y() << " " << p1.z() << "  "
	  << p2.x() << " " << p2.y() << " " << p2.z() << "  "
	  << p3.x() << " " << p3.y() << " " << p3.z() << "  "
	  << p4.x() << " " << p4.y() << " " << p4.z()
          << " ]\n";
    }
    saveinfo->end_attr(out);
  } else {
    for(int i=0;i<n;i++){
      saveinfo->start_attr(out);

      saveinfo->indent(out);
      out << "Color [ " << back_matls[i]->diffuse.r() << " "
	  << back_matls[i]->diffuse.g() << " "
	  << back_matls[i]->diffuse.b() << " ]\n";

      saveinfo->indent(out);
      out << "Surface \"plastic\" \"Ka\" 0.0 \"Kd\" 1.0 \"Ks\" 1.0 \"roughness\" "
	  << 1.0 / back_matls[i]->shininess << " \"specularcolor\" [ "
	  << back_matls[i]->specular.r() << " "
	  << back_matls[i]->specular.g() << " "
	  << back_matls[i]->specular.b() << " ]\n";

      Point from(positions[i]+directions[i]*headlength);

      Point p1(from+v1[i]);
      Point p2(from+v2[i]);
      Point p3(from-v1[i]);
      Point p4(from-v2[i]);
      
      saveinfo->indent(out);
      out << "Polygon \"P\" [ "
	  << p1.x() << " " << p1.y() << " " << p1.z() << "  "
	  << p2.x() << " " << p2.y() << " " << p2.z() << "  "
	  << p3.x() << " " << p3.y() << " " << p3.z() << "  "
	  << p4.x() << " " << p4.y() << " " << p4.z() << " ]\n";
      saveinfo->end_attr(out);
    }

  }

  // Draw the head.
  if(head_matls.size() == 1){
    // Color is same on each.

    saveinfo->start_attr(out);

    saveinfo->indent(out);
    out << "Color [ " << head_matls[0]->diffuse.r() << " "
	<< head_matls[0]->diffuse.g() << " "
	<< head_matls[0]->diffuse.b() << " ]\n";

    saveinfo->indent(out);
    out << "Surface \"plastic\" \"Ka\" 0.0 \"Kd\" 1.0 \"Ks\" 1.0 \"roughness\" "
	<< 1.0 / head_matls[0]->shininess << " \"specularcolor\" [ "
	<< head_matls[0]->specular.r() << " "
	<< head_matls[0]->specular.g() << " "
	<< head_matls[0]->specular.b() << " ]\n";

    for(int i=0;i<n;i++){
      Point from(positions[i]+directions[i]*headlength);
      Point top(positions[i]+directions[i]);

      Point p1(from+v1[i]);
      Point p2(from+v2[i]);
      Point p3(from-v1[i]);
      Point p4(from-v2[i]);
      
      saveinfo->indent(out);
      out << "Polygon \"P\" [ "
	  << top.x() << " " << top.y() << " " << top.z()
	  << p1.x() << " " << p1.y() << " " << p1.z() << "  "
	  << p2.x() << " " << p2.y() << " " << p2.z() << "  "
          << " ]\n";

      out << "Polygon \"P\" [ "
	  << top.x() << " " << top.y() << " " << top.z()
	  << p2.x() << " " << p2.y() << " " << p2.z() << "  "
	  << p3.x() << " " << p3.y() << " " << p3.z() << "  "
          << " ]\n";

      out << "Polygon \"P\" [ "
	  << top.x() << " " << top.y() << " " << top.z()
	  << p3.x() << " " << p3.y() << " " << p3.z() << "  "
	  << p4.x() << " " << p4.y() << " " << p4.z()
          << " ]\n";

      out << "Polygon \"P\" [ "
	  << top.x() << " " << top.y() << " " << top.z()
	  << p4.x() << " " << p4.y() << " " << p4.z()
	  << p1.x() << " " << p1.y() << " " << p1.z() << "  "
          << " ]\n";

    }
    saveinfo->end_attr(out);
  }
#if 0
 else {
    double w=headwidth;
    double h=1.0-headlength;
    double w2h2=w*w+h*h;
    for(int i=0;i<n;i++){
      saveinfo->start_attr(out);
      
      matl[i]->diffuse.get_color(color);
      saveinfo->indent(out);
      out << "Color [ " << color.r() << " " << color.g() << " " << color.b() << " ]\n";

      matl[i]->specular.get_color(color);
      saveinfo->indent(out);
      out << "Surface \"plastic\" \"Ka\" 0.0 \"Kd\" 1.0 \"Ks\" 1.0 \"roughness\" "
	  << 1.0 / matl[i]->shininess << " \"specularcolor\" [ "
	  << color.r() << " " << color.g() << " " << color.b() << " ]\n";
      
      glBegin(GL_TRIANGLES);
      Vector dn(directions[i]*w2h2);
      Vector n(dn+v1[i]+v2[i]);
      glNormal3d(n.x(), n.y(), n.z());
      di->set_matl(head_matls[i].get_rep());

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

} // End namespace SCIRun

// $Log

