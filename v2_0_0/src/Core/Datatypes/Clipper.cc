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

//    File   : BoxClipper.cc
//    Author : Michael Callahan
//    Date   : September 2001

#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Point.h>
#include <Core/Datatypes/Clipper.h>


namespace SCIRun {

PersistentTypeID Clipper::type_id("Clipper", "Datatype", 0);


Clipper::~Clipper()
{
}


bool
Clipper::inside_p(const Point &/* p */)
{
  return false;
}


const int CLIPPER_VERSION = 1;

void
Clipper::io(Piostream &stream)
{
  stream.begin_class("Clipper", CLIPPER_VERSION);
  stream.end_class();
}



PersistentTypeID IntersectionClipper::type_id("IntersectionClipper", "Clipper", 0);

IntersectionClipper::IntersectionClipper(ClipperHandle c0, ClipperHandle c1)
  : clipper0_(c0),
    clipper1_(c1)
{
}



bool
IntersectionClipper::inside_p(const Point &p)
{
  // Use boolean OR, is shortcutting.
  return !(!clipper0_->inside_p(p) || !clipper1_->inside_p(p));
}


const int INTERSECTIONCLIPPER_VERSION = 1;

void
IntersectionClipper::io(Piostream &stream)
{
  stream.begin_class("IntersectionClipper", INTERSECTIONCLIPPER_VERSION);
  Pio(stream, clipper0_);
  Pio(stream, clipper1_);
  stream.end_class();
}




PersistentTypeID UnionClipper::type_id("UnionClipper", "Clipper", 0);


UnionClipper::UnionClipper(ClipperHandle c0, ClipperHandle c1)
  : clipper0_(c0),
    clipper1_(c1)
{
}


bool
UnionClipper::inside_p(const Point &p)
{
  return clipper0_->inside_p(p) || clipper1_->inside_p(p);
}


const int UNIONCLIPPER_VERSION = 1;

void
UnionClipper::io(Piostream &stream)
{
  stream.begin_class("UnionClipper", UNIONCLIPPER_VERSION);
  Pio(stream, clipper0_);
  Pio(stream, clipper1_);
  stream.end_class();
}




PersistentTypeID InvertClipper::type_id("InvertClipper", "Clipper", 0);

InvertClipper::InvertClipper(ClipperHandle c)
  : clipper_(c)
{
}


bool
InvertClipper::inside_p(const Point &p)
{
  return !clipper_->inside_p(p);
}


const int INVERTCLIPPER_VERSION = 1;

void
InvertClipper::io(Piostream &stream)
{
  stream.begin_class("InvertClipper", INVERTCLIPPER_VERSION);
  Pio(stream, clipper_);
  stream.end_class();
}




PersistentTypeID BoxClipper::type_id("BoxClipper", "Clipper", 0);

BoxClipper::BoxClipper(Transform &t)
  : trans_(t)
{
}


bool
BoxClipper::inside_p(const Point &p)
{
  Point ptrans;
  ptrans = trans_.project(p);
  return (ptrans.x() >= -1.0 && ptrans.x() < 1.0 &&
	  ptrans.y() >= -1.0 && ptrans.y() < 1.0 &&
	  ptrans.z() >= -1.0 && ptrans.z() < 1.0);
}


const int BOXCLIPPER_VERSION = 2;

void
BoxClipper::io(Piostream &stream)
{
  int version = stream.begin_class("BoxClipper", BOXCLIPPER_VERSION);
  if (version < 2 && stream.reading() ) {
    Pio_old(stream, trans_);
  } else {
    Pio(stream, trans_);
  }
  stream.end_class();
}





} // end namespace SCIRun

