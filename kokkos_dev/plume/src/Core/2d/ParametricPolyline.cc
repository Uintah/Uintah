/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

#include <iostream>
#include <sstream>

using namespace std;

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

  // THIS pio seems to be undefined.
  //  Pio(stream, data_);
  cout << "fix the above pio in ParametricPolyline.cc.  Bye.\n";
  exit(1);

  Pio(stream, tmin_);
  Pio(stream, tmax_);
  Pio(stream, xmin_);
  Pio(stream, xmax_);
  Pio(stream, ymin_);
  Pio(stream, ymax_);
  stream.end_class();
}


} // namespace SCIRun

  

