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

#include <Core/Datatypes/Color.h>
#include <Core/Malloc/Allocator.h>
#include <Core/2d/Hairline.h>
#include <Core/2d/Diagram.h>
#include <Core/Containers/Array1.h>
#include <Core/2d/LockedPolyline.h>
#include <Core/GuiInterface/GuiInterface.h>

using namespace SCIRun;
Persistent* make_Hairline()
{
  return scinew Hairline(GuiInterface::getSingleton());
}

PersistentTypeID Hairline::type_id("Hairline", "HairObj", make_Hairline);

Hairline::Hairline(GuiInterface* gui, Diagram *p, const string &name)
  : TclObj(gui, "Hairline" ), HairObj(name), parent_(p)
{
}


Hairline::~Hairline()
{
}


void
Hairline::select( double x, double y, int b )
{
  HairObj::select( x, y, b );
}
  
void
Hairline::move( double x, double y, int b)
{
  HairObj::move( x, y, b );
}
  
void
Hairline::release( double x, double y, int b)
{
  HairObj::release( x, y, b );
}
  

void
Hairline::update()
{
  parent_->get_active( poly_ );

  tcl_ << " values ";

  if ( poly_.size() > 0 ) {
    
    double pos = parent_->x_get_at( at() );
    
    // get and sort the values
    
    Array1<int> index(poly_.size());
    Array1<double> value(poly_.size());
    
    // get value and insert them in acsending order (using insert sort)
    for (int i=0; i<poly_.size(); i++) {
      if (LockedPolyline *p = dynamic_cast<LockedPolyline*>(poly_[i])) {
        double v = p->at(pos);
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
    }
  
    // send the sorted list to the tcl side
    for (int i=0; i<poly_.size(); i++) 
      tcl_ << " { " << poly_[index[i]]->tcl_color()
	   << " " << value[i] << " } ";
  }

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
