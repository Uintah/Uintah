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


#include <stdio.h>

#include <Core/Datatypes/Color.h>
#include <Core/Malloc/Allocator.h>
#include <Core/2d/Axes.h>
#include <Core/2d/Diagram.h>
#include <Core/GuiInterface/GuiInterface.h>

using namespace SCIRun;

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
