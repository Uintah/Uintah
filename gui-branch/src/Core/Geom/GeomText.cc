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
 *  GeomText.cc:  Texts of GeomObj's
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Mar 1998
 *
 *  Copyright (C) 1998 SCI Text
 */

#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geom/Color.h>
#include <Core/Geometry/Point.h>
#include <Core/Geom/GeomSave.h>
#include <Core/Geom/GeomText.h>
#include <iostream>
using std::ostream;

namespace SCIRun {

int    GeomText::init = 1;
GLuint GeomText::fontbase = 0;

static Persistent* make_GeomText()
{
    return scinew GeomText;
}

PersistentTypeID GeomText::type_id("GeomText", "GeomObj", make_GeomText);

GeomText::GeomText()
  : GeomObj()
{
}

GeomText::GeomText( const string &text, const Point &at, const Color &c)
: GeomObj(), text(text), at(at), c(c)
{
}

GeomText::GeomText(const GeomText& copy)
: GeomObj(copy)
{
  init = copy.init;
  fontbase = copy.fontbase;
  text = copy.text;
  at = copy.at;
  c = copy.c;
}


GeomObj* GeomText::clone()
{
    return scinew GeomText(*this);
}

void GeomText::get_bounds(BBox& in_bb)
{
  in_bb.extend( at );
}

GeomText::~GeomText()
{
}

void GeomText::reset_bbox()
{
}

#define GEOMGROUP_VERSION 1

void GeomText::io(Piostream& stream)
{

    stream.begin_class("GeomText", GEOMGROUP_VERSION);
    // Do the base class first...
    GeomObj::io(stream);
    Pio(stream, at);
    Pio(stream, text);
    stream.end_class();
}

bool GeomText::saveobj(ostream&, const string&, GeomSave*)
{
  return 0;
}


} // End namespace SCIRun


