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

#include <Core/Malloc/Allocator.h>
#include <Core/2d/Hairline.h>


namespace SCIRun {

Persistent* make_Hairline()
{
  return scinew Hairline;
}

PersistentTypeID Hairline::type_id("Hairline", "Widget", make_Hairline);

Hairline::Hairline( const BBox2d &bbox, const string &name)
  : TclObj( name ), Widget(name), hair_(0)
{
  hair_ = scinew HairObj( bbox, "hair" );
}


Hairline::~Hairline()
{
}


void 
Hairline::add( Polyline *p )
{
  poly_.add( p );
  Color c = p->get_color();
  tcl_ << " add " << (poly_.size()-1) << " #";
  tcl_.setf(ios::hex,ios::basefield);
  tcl_<< int(c.r()*255) << int(c.g()*255) << int(c.b()*255) << " "
       << p->at( hair_->at() );
  tcl_exec();
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
  cerr << "release\n";
  update();
}
  

void
Hairline::update()
{
  tcl_ << " values ";
  double at = hair_->at();
  for (int i=0; i<poly_.size(); i++) 
    tcl_ << poly_[i]->at( at ) << " ";
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

  
