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
#include <Core/Persistent/PersistentSTL.h>
#include <Core/2d/Polyline.h>
#include <Core/2d/BBox2d.h>

#include <iostream>
#include <sstream>

using namespace std;

namespace SCIRun {

Persistent* make_Polyline()
{
  return scinew Polyline;
}

PersistentTypeID Polyline::type_id("Polyline", "DrawObj", make_Polyline);


Polyline::Polyline( const vector<double> &data, const string &name )
  : DrawObj(name), data_(data) 
{
  compute_minmax();
  color_ = Color( 0, 0, 0 );
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
Polyline::compute_minmax()
{
  if ( data_.size() > 0 ) {
    min_ = max_ = data_[0];
    for (unsigned i=1; i<data_.size(); i++)
      if ( data_[i] < min_ ) min_ = data_[i];
      else if ( max_ < data_[i] ) max_ = data_[i];
  }
}


void
Polyline::set_color( const Color &c )
{
  color_ = c;
}

void
Polyline::add( const vector<double>& v)
{
  for (unsigned i=0; i<v.size(); ++i)
    data_.push_back(v[i]);    
 
  compute_minmax();
}

void
Polyline::add( double v )
{
  if ( data_.size() == 0 ) 
    min_ = max_ = v;
  else
    if ( v < min_ ) min_ = v;
    else if ( max_ < v ) max_ = v;

  data_.push_back(v);
}

double
Polyline::at( double v )
{
  double val;

  if ( v < 0 ) 
    val = data_[0];
  else if ( v >= data_.size()-1 ) 
    val = data_[data_.size()-1];
  else {
    int p = int(v);
    val  = data_[p] + (data_[p+1] - data_[p])*(v-p);
  }

  return val;
}

void
Polyline::get_bounds( BBox2d &bb )
{
  bb.extend( Point2d(0, min_));
  bb.extend( Point2d(data_.size()-1, max_ ) );
}

#define POLYLINE_VERSION 1

void 
Polyline::io(Piostream& stream)
{
  stream.begin_class("Polyline", POLYLINE_VERSION);
  DrawObj::io(stream);
  Pio(stream, data_);
  Pio(stream, min_);
  Pio(stream, max_);
  stream.end_class();
}


} // namespace SCIRun

  
