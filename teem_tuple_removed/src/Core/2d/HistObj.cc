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

  
