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
 *  ParametricPolyline.cc: Displayable 2D object
 *
 *  Written by:
 *   Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Malloc/Allocator.h>
#include <Core/2d/ParametricPolyline.h>
#include <Core/2d/BBox2d.h>

#include <stdio.h>

namespace SCIRun {

Persistent* make_ParametricPolyline()
{
  return scinew ParametricPolyline;
}

PersistentTypeID ParametricPolyline::type_id("ParametricPolyline", "DrawObj", make_ParametricPolyline);


ParametricPolyline::ParametricPolyline( const map<double, pair<double, double> > &data, const string &name )
  : DrawObj(name), data_(data) 
{
  compute_minmax();
  color_ = Color( 0, 0, 0 );
}


ParametricPolyline::ParametricPolyline( int i )
{
  ostringstream name;
  
  name << "line-"<<i;
  set_name( name.str() );
}
  
  
ParametricPolyline::~ParametricPolyline()
{
}

void
ParametricPolyline::compute_minmax()
{
  iter i = data_.begin();
  
  if (i != data_.end()) {
    tmin_ = (*i).first;
    xmin_ = (*i).second.first;  
    ymin_ = (*i).second.second;
    tmax_ = (*i).first;
    xmax_ = (*i).second.first;
    ymax_ = (*i).second.second;
  } else
    return;

  ++i;

  while (i != data_.end()) {
    ++i;
    if ((*i).first < tmin_) 
      tmin_ = (*i).first;
    else if ((*i).first > tmax_)
      tmax_ = (*i).first;

    if ((*i).second.first < xmin_) 
      xmin_ = (*i).second.first;
    else if ((*i).second.first > xmax_) 
      xmax_ = (*i).second.first;

    if ((*i).second.second < ymin_) 
      ymin_ = (*i).second.second;
    else if ((*i).second.second > ymax_) 
      ymax_ = (*i).second.second;
        
  }
}

void
ParametricPolyline::set_color( const Color &c )
{
  color_ = c;
}

void
ParametricPolyline::add(const vector<double>& v)
{
  for (unsigned i=0; i<v.size()-2; i+=3) 
    data_.insert(pair<double, 
      pair<double,double> >(v[i],pair<double,double>(v[i+1],v[i+2])));

  compute_minmax();
}

void
ParametricPolyline::add( double t, double x, double y)
{
  if ( data_.size() == 0 ) {
    tmin_ = tmax_ = t;
    xmin_ = xmax_ = x;
    ymin_ = ymax_ = y;
  } else {
    if ( t < tmin_) 
      tmin_ = t;
    else if ( t > tmax_ ) 
      tmax_ = t;

    if ( x < xmin_) 
      xmin_ = x;
    else if ( x > xmax_ ) 
      xmax_ = x;

    if ( y < ymin_) 
      ymin_ = y;
    else if ( y > ymax_ ) 
      ymax_ = y;
  }

  data_.insert(pair<double, 
                    pair<double,double> >(t,pair<double,double>(x,y)));
}

bool
ParametricPolyline::at( double v, pair<double,double>& d )
{
  iter i = data_.find(v);

  if (i == data_.end())
    return false;
      
  d.first = (*i).second.first;
  d.second = (*i).second.second;

  return true;
}

void
ParametricPolyline::get_bounds( BBox2d &bb )
{
  bb.extend( Point2d(xmin_, ymin_) );
  bb.extend( Point2d(xmax_, ymax_) );
}

#define PARAMETRICPOLYLINE_VERSION 1

void 
ParametricPolyline::io(Piostream& stream)
{
  stream.begin_class("ParametricPolyline", PARAMETRICPOLYLINE_VERSION);
  DrawObj::io(stream);
  Pio(stream, data_);
  Pio(stream, tmin_);
  Pio(stream, tmax_);
  Pio(stream, xmin_);
  Pio(stream, xmax_);
  Pio(stream, ymin_);
  Pio(stream, ymax_);
  stream.end_class();
}


} // namespace SCIRun

  

