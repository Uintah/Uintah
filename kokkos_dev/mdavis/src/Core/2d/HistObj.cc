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
 *  HistObj.cc: Displayable 2D object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <stdio.h>

#include <Core/Malloc/Allocator.h>
#include <Core/2d/HistObj.h>
#include <Core/Persistent/PersistentSTL.h>

namespace SCIRun {

Persistent* make_HistObj()
{
  return scinew HistObj;
}

PersistentTypeID HistObj::type_id("HistObj", "DrawObj", make_HistObj);


HistObj::HistObj( const string &name)
  :Polyline(name), bins_(128)
{
}


HistObj::HistObj( const Array1<double> &data, const string &name )
  : Polyline(name), ref_(data), bins_(128)
{
  set_data( data );
}

HistObj::~HistObj()
{
}

void
HistObj::set_data( const Array1<double> & data )
{
  ref_ = data;
  if ( ref_.size() > 0 ) {
    min_ = max_ = ref_[0];
    for (int i=1; i<ref_.size(); i++)
      if ( ref_[i] < min_ ) min_ = ref_[i];
      else if ( max_ < ref_[i] ) max_ = ref_[i];
  }
  compute();
}

void
HistObj::set_bins( int n )
{
  if ( bins_ == n ) 
    return;

  bins_ = n;
  compute();
}

void
HistObj::compute()
{
  data_.resize( bins_ );
  for (int i=0; i<bins_; i++)
    data_[i] = 0;

  ref_min_ = ref_max_ = ref_[0];
  for (int i=0; i<ref_.size(); i++) {
    // compute ref_ min/max
    if ( ref_[i] < ref_min_ ) ref_min_ = ref_[i];
    else if ( ref_max_ < ref_[i] ) ref_max_ = ref_[i];

    // histogram 
    int pos = int((ref_[i]-min_) * bins_ / ( max_ - min_ ));
    if ( pos == bins_ ) pos--;
    data_[pos]++;
  }

  for (unsigned int i=0; i<data_.size(); i++)
    data_[i] /= ref_.size();

  compute_minmax();
}
  

double
HistObj::at( double v )
{
  double val;

  if ( v < ref_min_  || v > ref_max_ ) 
    val = 0;
  else {
    int pos ((int)(bins_ * (v-ref_min_)/ (ref_max_ - ref_min_)) );
    if ( pos == (int)(data_.size()) ) 
      pos--;
    val = data_[pos];
  }

  return val;
}


void
HistObj::get_bounds( BBox2d &bb )
{
  bb.extend( Point2d(ref_min_, min_));
  bb.extend( Point2d(ref_max_, max_ ) );
}


#define HistObj_VERSION 1

void 
HistObj::io(Piostream& stream)
{
  stream.begin_class("HistObj", HistObj_VERSION);
  DrawObj::io(stream);
  Pio(stream, data_);
  Pio(stream, min_);
  Pio(stream, max_);
  stream.end_class();
}


} // namespace SCIRun

  
