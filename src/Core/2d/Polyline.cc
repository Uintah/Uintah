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
 *  Polyline.cc: Displayable 2D object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Malloc/Allocator.h>
#include <Core/2d/Polyline.h>
#include <Core/2d/BBox2d.h>
#include <iostream>
using std::cerr;
using std::ostream;
#include <sstream>
using std::ostringstream;

namespace SCIRun {

Persistent* make_Polyline()
{
  return scinew Polyline;
}

PersistentTypeID Polyline::type_id("polyline", "Drawable", make_Polyline);


Polyline::Polyline( const Array1<double> &data, const string &name )
  : Drawable(name), data_(data) 
{
  if ( data_.size() > 0 ) {
    min_ = max_ = data_[0];
    for (int i=1; i<data_.size(); i++)
      if ( data_[i] < min_ ) min_ = data_[i];
      else if ( max_ < data_[i] ) max_ = data_[i];
  }

  color = Color( 0, 0, 0 );
}


Polyline::Polyline( int i )
{
  ostringstream name;
  
  name << "line-"<<i;
  set_name( name.str() );
}
  
  
Polyline::~Polyline()
{
}

void
Polyline::set_color( const Color &c )
{
  color = c;
}

void
Polyline::add( double v )
{
  if ( data_.size() == 0 ) 
    min_ = max_ = v;
  else
    if ( v < min_ ) min_ = v;
    else if ( max_ < v ) max_ = v;

  lock();
  data_.add(v);
  unlock();
}

void
Polyline::get_bounds( BBox2d &bb )
{
  lock();
  bb.extend( Point2d(0, min_));
  bb.extend( Point2d(data_.size()-1, max_ ) );
  unlock();
}

#define POLYLINE_VERSION 1

void 
Polyline::io(Piostream& stream)
{
  stream.begin_class("Polyline", POLYLINE_VERSION);
  Drawable::io(stream);
  Pio(stream, data_);
  Pio(stream, min_);
  Pio(stream, max_);
  stream.end_class();
}


} // namespace SCIRun

  
