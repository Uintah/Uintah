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

PersistentTypeID Hairline::type_id("Hairline", "HairObj", make_Hairline);

Hairline::Hairline( Diagram *p, const string &name)
  : TclObj( "Hairline" ), HairObj(name), parent_(p)
{
}


Hairline::~Hairline()
{
}


void
Hairline::select( double x, double y, int b )
{
  HairObj::select( x, y, b );
  //  update();
}
  
void
Hairline::move( double x, double y, int b)
{
  HairObj::move( x, y, b );
  //  update();
}
  
void
Hairline::release( double x, double y, int b)
{
  HairObj::release( x, y, b );
  //  update();
}
  

void
Hairline::update()
{
  parent_->get_active( poly_ );
  double pos = parent_->get_at( at() );

  // get and sort the values

  int index[poly_.size()];
  double value[poly_.size()];

  // get value and insert them in acsending order (using insert sort)
  for (int i=0; i<poly_.size(); i++) {
    double v = poly_[i]->at(pos);
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

  
