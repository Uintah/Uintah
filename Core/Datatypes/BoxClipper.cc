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
#include <Core/Datatypes/BoxClipper.h>


namespace SCIRun {


Clipper::~Clipper()
{
}


bool
Clipper::inside_p(const Point &/* p */)
{
  return false;
}



PersistentTypeID BoxClipper::type_id("BoxClipper", "Clipper", 0);


BoxClipper::BoxClipper(Transform &t)
  : trans_(t)
{
}


BoxClipper::BoxClipper(const BoxClipper &bc)
  : trans_(bc.trans_)
{
}


BoxClipper::~BoxClipper()
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


const int BOXCLIPPER_VERSION = 1;

void
BoxClipper::io(Piostream &stream)
{
  stream.begin_class("BoxClipper", BOXCLIPPER_VERSION);
  Pio(stream, trans_);
  stream.end_class();
}


} // end namespace SCIRun

