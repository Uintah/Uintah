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
#include <Core/Datatypes/Color.h>
#include <Core/Geometry/Point.h>
#include <Core/Geom/GeomSave.h>
#include <Core/Geom/GeomText.h>
#include <Core/Persistent/PersistentSTL.h>
#include <iostream>

namespace SCIRun {

using std::ostream;

static Persistent* make_GeomText()
{
    return scinew GeomText;
}

PersistentTypeID GeomText::type_id("GeomText", "GeomObj", make_GeomText);

GeomText::GeomText()
  : GeomObj()
{
}

GeomText::GeomText( const string &text, const Point &at, const Color &c,
		    const string &fontsize)
  : GeomObj(), text(text), fontsize(fontsize), at(at), c(c)
{
}

GeomText::GeomText(const GeomText& copy)
: GeomObj(copy)
{
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

#define GEOMTEXT_VERSION 2

void
GeomText::io(Piostream& stream)
{

    const int version = stream.begin_class("GeomText", GEOMTEXT_VERSION);
    // Do the base class first...
    GeomObj::io(stream);
    Pio(stream, at);
    Pio(stream, text);
    if (version > 1)
    {
      Pio(stream, c);
    }
    stream.end_class();
}

static Persistent* make_GeomTexts()
{
    return scinew GeomTexts;
}

PersistentTypeID GeomTexts::type_id("GeomTexts", "GeomObj", make_GeomTexts);

GeomTexts::GeomTexts()
  : GeomObj(),
    fontindex_(2)
{
}


GeomTexts::GeomTexts(const GeomTexts& copy) :
  GeomObj(copy),
  fontindex_(copy.fontindex_),
  text_(copy.text_),
  location_(copy.location_),
  color_(copy.color_)
{
}


GeomTexts::~GeomTexts()
{
}


GeomObj* GeomTexts::clone()
{
    return scinew GeomTexts(*this);
}


void
GeomTexts::get_bounds(BBox& in_bb)
{
  for (unsigned int i = 0; i < location_.size(); i++)
  {
    in_bb.extend( location_[i] );
  }
}

void
GeomTexts::reset_bbox()
{
}


void
GeomTexts::set_font_index(int a)
{
  if (a >= 0 && a < 5)
  {
    fontindex_ = a;
  }
}


void
GeomTexts::add(const string &t, const Point &p)
{
  text_.push_back(t);
  location_.push_back(p);
}

void
GeomTexts::add(const string &t, const Point &p, const Color &c)
{
  add(t, p);
  color_.push_back(c);
}

void
GeomTexts::add(const string &t, const Point &p, float index)
{
  add(t, p);
  index_.push_back(index);
}


#define GEOMTEXTS_VERSION 2

void
GeomTexts::io(Piostream& stream)
{
    const int version = stream.begin_class("GeomTexts", GEOMTEXTS_VERSION);
    // Do the base class first.
    GeomObj::io(stream);
    Pio(stream, fontindex_);
    Pio(stream, text_);
    Pio(stream, location_);
    Pio(stream, color_);
    if (version > 1) { Pio(stream, index_); }
    stream.end_class();
}

static Persistent* make_GeomTextsCulled()
{
    return scinew GeomTextsCulled;
}

PersistentTypeID GeomTextsCulled::type_id("GeomTextsCulled", "GeomTexts", make_GeomTextsCulled);


GeomTextsCulled::GeomTextsCulled()
  : GeomTexts()
{
}


GeomTextsCulled::GeomTextsCulled(const GeomTextsCulled& copy) :
  GeomTexts(copy),
  normal_(copy.normal_)
{
}


GeomTextsCulled::~GeomTextsCulled()
{
}


GeomObj* GeomTextsCulled::clone()
{
    return scinew GeomTextsCulled(*this);
}


void
GeomTextsCulled::add(const string &t, const Point &p, const Vector &n)
{
  GeomTexts::add(t, p);
  normal_.push_back(n);
}


void
GeomTextsCulled::add(const string &t, const Point &p,
		     const Vector &n, const Color &c)
{
  GeomTexts::add(t, p, c);
  normal_.push_back(n);
}


void
GeomTextsCulled::add(const string &t, const Point &p,
		     const Vector &n, float index)
{
  GeomTexts::add(t, p, index);
  normal_.push_back(n);
}


#define GEOMCULLEDTEXTS_VERSION 1

void
GeomTextsCulled::io(Piostream& stream)
{
    stream.begin_class("GeomTextsCulled", GEOMCULLEDTEXTS_VERSION);
    // Do the base class first...
    GeomTexts::io(stream);
    Pio(stream, normal_);
    stream.end_class();
}


} // End namespace SCIRun


