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
  Vector vv1, vv2;
  if(!dir.check_find_orthogonal(vv1, vv2))
    return;

  positions.add(pos);
  directions.add(dir);
  if (!normalize_headsize) {
    // use the length to scale the head
    double len = dir.length();
    v1.add(vv1*headwidth*len);
    v2.add(vv2*headwidth*len);
  } else {
    // don't scale the head by the length
    vv1.normalize();
    vv2.normalize();
    v1.add(vv1*headwidth);
    v2.add(vv2*headwidth);
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


} // End namespace SCIRun

// $Log

