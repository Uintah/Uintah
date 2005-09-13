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
 *  Axes.cc: 
 *
 *  Written by:
 *   Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   Aug 2001
 *
 *  Copyright (C) 2001 SCI Group
 */


#include <Core/Datatypes/Color.h>
#include <Core/Malloc/Allocator.h>
#include <Core/2d/Axes.h>
#include <Core/2d/Diagram.h>
#include <Core/GuiInterface/GuiInterface.h>

#include <iostream>

using namespace SCIRun;
using namespace std;

Persistent* make_XAxis()
{
  return scinew XAxis(GuiInterface::getSingleton());
}

PersistentTypeID XAxis::type_id("XAxis", "XAxisObj", make_XAxis);

XAxis::XAxis(GuiInterface* gui, Diagram *p, const string &name)
  : TclObj(gui, "XAxis" ), XAxisObj(name), parent_(p), activepoly_(0),
    initialized_(false)
{
}


XAxis::~XAxis()
{
}


void
XAxis::select( double x, double y, int b )
{
  cerr << "X Axis select\n";
  XAxisObj::select( x, y, b );
  //  update();
}
  
void
XAxis::move( double x, double y, int b)
{
  XAxisObj::move( x, y, b );
  //  update();
}
  
void
XAxis::release( double x, double y, int b)
{
  XAxisObj::release( x, y, b );
  //  update();
}
  

void
XAxis::update()
{
}


#define XAXIS_VERSION 1

void 
XAxis::io(Piostream& stream)
{
  stream.begin_class("XAxis", XAXIS_VERSION);
  Widget::io(stream);
  stream.end_class();
}

Persistent* make_YAxis()
{
  return scinew YAxis(GuiInterface::getSingleton());;
}

PersistentTypeID YAxis::type_id("YAxis", "YAxisObj", make_YAxis);

YAxis::YAxis(GuiInterface* gui, Diagram *p, const string &name)
  : TclObj(gui, "YAxis" ), YAxisObj(name), parent_(p), activepoly_(0),
    initialized_(false)
{
}


YAxis::~YAxis()
{
}


void
YAxis::select( double x, double y, int b )
{
  cerr << "Y Axis select\n";
  YAxisObj::select( x, y, b );
  //  update();
}
  
void
YAxis::move( double x, double y, int b)
{
  YAxisObj::move( x, y, b );
  //  update();
}
  
void
YAxis::release( double x, double y, int b)
{
  YAxisObj::release( x, y, b );
  //  update();
}
  

void
YAxis::update()
{
}


#define YAXIS_VERSION 1

void 
YAxis::io(Piostream& stream)
{
  stream.begin_class("YAxis", YAXIS_VERSION);
  Widget::io(stream);
  stream.end_class();
}
