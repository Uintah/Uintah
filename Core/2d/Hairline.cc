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
 *  Hairline.cc: 
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Aug 2001
 *
 *  Copyright (C) 2001 SCI Group
 */


#include <stdio.h>
#include <iostream>
using std::cerr;
using std::ostream;
#include <sstream>
using std::ostringstream;

#include <Core/Geom/Color.h>
#include <Core/Malloc/Allocator.h>
#include <Core/2d/Hairline.h>
#include <Core/2d/Diagram.h>

namespace SCIRun {

Persistent* make_Hairline()
{
  return scinew Hairline;
}

PersistentTypeID Hairline::type_id("Hairline", "Widget", make_Hairline);

Hairline::Hairline( Diagram *p, const string &name)
  : TclObj( "Hairline" ), Widget(name), hair_(0), parent_(p)
{
  parent_->get_active( poly_ );

  BBox2d bb;
  for (int i=0; i<poly_.size(); i++) 
    poly_[i]->get_bounds( bb );
  
  bbox_.extend( Point2d( bb.min().x(), 0 ) );
  bbox_.extend( Point2d( bb.max().x(), 1 ) );

  hair_ = scinew HairObj( bbox_, "hair" );
}


Hairline::~Hairline()
{
}


void
Hairline::select( double x, double y, int b )
{
  hair_->select( x, y, b );
  update();
}
  
void
Hairline::move( double x, double y, int b)
{
  hair_->move( x, y, b );
  update();
}
  
void
Hairline::release( double x, double y, int b)
{
  hair_->release( x, y, b );
  update();
}
  

void
Hairline::update()
{
  parent_->get_active( poly_ );

  // resert hair postion

  bbox_.reset();
  for (int i=0; i<poly_.size(); i++) 
    poly_[i]->get_bounds( bbox_ );

  if ( !bbox_.valid() )
    return;

  hair_->set_bbox ( BBox2d( Point2d( bbox_.min().x(), 0 ),
			    Point2d( bbox_.max().x(), 1 ) ) );

  // get and sort the values

  double at = hair_->at();
  int index[poly_.size()];
  double value[poly_.size()];

  // get value and insert them in acsending order (using insert sort)
  for (int i=0; i<poly_.size(); i++) {
    double v = poly_[i]->at(at);
    int j;
    for (j=i; j>0; j--) {
      if ( value[j-1] < v ) {
	value[j] = value[j-1];
	index[j] = index[j-1];
      }
      else 
	break;
    }
    value[j] = v;
    index[j] = i;
  }

  // send the sorted list to the tcl side
  tcl_ << " values ";
  for (int i=0; i<poly_.size(); i++) 
    tcl_ << " { " << poly_[index[i]]->tcl_color()
	 << " " << value[i] << " } ";
  tcl_exec();
}


#define HAIRLINE_VERSION 1

void 
Hairline::io(Piostream& stream)
{
  stream.begin_class("Hairline", HAIRLINE_VERSION);
  Widget::io(stream);
  stream.end_class();
}


} // namespace SCIRun

  
