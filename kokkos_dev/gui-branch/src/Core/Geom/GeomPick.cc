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
 *  Pick.cc: Picking information for Geometry objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/GeomPick.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::ostream;

namespace SCIRun {

Persistent* make_GeomPick()
{
    return scinew GeomPick(0,0);
}

PersistentTypeID GeomPick::type_id("GeomPick", "GeomObj", make_GeomPick);

GeomPick::GeomPick(GeomObj* obj, ModulePickable* module)
: GeomContainer(obj),
  module(module),
  cbdata(0),
  picked_obj(0),
  directions(6),
  widget(0),
  selected(0),
  ignore(0),
  drawOnlyOnPick(0)
{
    directions[0]=Vector(1,0,0);
    directions[1]=Vector(-1,0,0);
    directions[2]=Vector(0,1,0);
    directions[3]=Vector(0,-1,0);
    directions[4]=Vector(0,0,1);
    directions[5]=Vector(0,0,-1);
}

GeomPick::GeomPick(GeomObj* obj, ModulePickable* module,
		   WidgetPickable* widget, int widget_data)
: GeomContainer(obj),
  module(module),
  cbdata(0),
  picked_obj(0),
  directions(6),
  widget(widget),
  widget_data(widget_data),
  selected(0),
  ignore(0),
  drawOnlyOnPick(0)
{
    directions[0]=Vector(1,0,0);
    directions[1]=Vector(-1,0,0);
    directions[2]=Vector(0,1,0);
    directions[3]=Vector(0,-1,0);
    directions[4]=Vector(0,0,1);
    directions[5]=Vector(0,0,-1);
}

GeomPick::GeomPick(GeomObj* obj, ModulePickable* module, const Vector& v1)
: GeomContainer(obj),
  module(module),
  cbdata(0),
  picked_obj(0),
  directions(2),
  widget(0),
  selected(0),
  ignore(0),
  drawOnlyOnPick(0)
{
    directions[0]=v1;
    directions[1]=-v1;
}

GeomPick::GeomPick(GeomObj* obj, ModulePickable* module, const Vector& v1, const Vector& v2)
: GeomContainer(obj),
  module(module),
  cbdata(0),
  picked_obj(0),
  directions(4),
  widget(0),
  selected(0),
  ignore(0),
  drawOnlyOnPick(0)
{
    directions[0]=v1;
    directions[1]=-v1;
    directions[2]=v2;
    directions[3]=-v2;
}

GeomPick::GeomPick(GeomObj* obj, ModulePickable* module, const Vector& v1, const Vector& v2,
		   const Vector& v3)
: GeomContainer(obj),
  module(module),
  cbdata(0),
  picked_obj(0),
  directions(6),
  widget(0),
  selected(0),
  ignore(0),
  drawOnlyOnPick(0)
{
    directions[0]=v1;
    directions[1]=-v1;
    directions[2]=v2;
    directions[3]=-v2;
    directions[4]=v3;
    directions[5]=-v3;
}

GeomPick::GeomPick(const GeomPick& copy)
: GeomContainer(copy),
  module(copy.module),
  cbdata(copy.cbdata), 
  picked_obj(copy.picked_obj),
  directions(copy.directions),
  widget(copy.widget),
  selected(copy.selected),
  ignore(copy.ignore),
  highlight(copy.highlight),
  drawOnlyOnPick(0)
{
}

GeomObj* GeomPick::clone()
{
    return scinew GeomPick(*this);
}

GeomPick::~GeomPick()
{
}

void GeomPick::set_highlight(const MaterialHandle& matl)
{
    highlight=matl;
}

void GeomPick::set_module_data(void* _cbdata)
{
    cbdata=_cbdata;
}

void GeomPick::set_widget_data(int _wd)
{
    widget_data=_wd;
}

void GeomPick::set_picked_obj(GeomObj* _object)
{
   picked_obj = _object;
}

void GeomPick::ignore_until_release()
{
    ignore=1;
}

void GeomPick::pick(ViewWindow* viewwindow, const BState& bs )
{
  selected=1;
  ignore=0;
  if(widget)
    widget->geom_pick(this, viewwindow, widget_data, bs);
  if(module)
    module->geom_pick(this, cbdata, picked_obj);
}

void GeomPick::release(const BState& bs)
{
    selected=0;
    if(widget)
	widget->geom_release(this, widget_data, bs);
    if(module)
	module->geom_release(this, cbdata, picked_obj);
}

void GeomPick::moved(int axis, double distance, const Vector& delta, const BState& bs)
{
    if(ignore) return;
    if(widget)
	widget->geom_moved(this, axis, distance, delta, widget_data, bs);
    if(module)
      module->geom_moved(this, axis, distance, delta, cbdata,picked_obj);
}

int GeomPick::nprincipal() {
    return directions.size();
}

Vector GeomPick::principal(int i) {
    return directions[i];
}

void GeomPick::set_principal(const Vector& v1)
{
    directions.remove_all();
    directions.grow(2);
    directions[0]=v1;
    directions[1]=-v1;
}

void GeomPick::set_principal(const Vector& v1, const Vector& v2)
{
    directions.remove_all();
    directions.grow(4);
    directions[0]=v1;
    directions[1]=-v1;
    directions[2]=v2;
    directions[3]=-v2;
}

void GeomPick::set_principal(const Vector& v1, const Vector& v2,
			     const Vector& v3)
{
    directions.remove_all();
    directions.grow(6);
    directions[0]=v1;
    directions[1]=-v1;
    directions[2]=v2;
    directions[3]=-v2;
    directions[4]=v3;
    directions[5]=-v3;
}

#define GEOMPICK_VERSION 1

void GeomPick::io(Piostream& stream)
{
    stream.begin_class("GeomPick", GEOMPICK_VERSION);
    GeomContainer::io(stream);
    stream.end_class();
}

bool GeomPick::saveobj(ostream& out, const string& format, GeomSave* saveinfo)
{
    return child->saveobj(out, format, saveinfo);
}

} // End namespace SCIRun

