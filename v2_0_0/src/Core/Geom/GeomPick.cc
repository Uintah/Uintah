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
#include <Core/GeomInterface/Pickable.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::ostream;

namespace SCIRun {

Persistent* make_GeomPick()
{
  return scinew GeomPick(0,0);
}


PersistentTypeID GeomPick::type_id("GeomPick", "GeomObj", make_GeomPick);


GeomPick::GeomPick(GeomHandle obj, ModulePickable* module)
  : GeomContainer(obj),
    module_(module),
    cbdata_(0),
    picked_obj_(0),
    directions_(6),
    widget_(0),
    selected_(false),
    ignore_(false),
    draw_only_on_pick_(false)
{
  directions_[0]=Vector(1,0,0);
  directions_[1]=Vector(-1,0,0);
  directions_[2]=Vector(0,1,0);
  directions_[3]=Vector(0,-1,0);
  directions_[4]=Vector(0,0,1);
  directions_[5]=Vector(0,0,-1);
}


GeomPick::GeomPick(GeomHandle obj, ModulePickable* module,
		   WidgetPickable* widget, int widget_data)
  : GeomContainer(obj),
    module_(module),
    cbdata_(0),
    picked_obj_(0),
    directions_(6),
    widget_(widget),
    widget_data_(widget_data),
    selected_(false),
    ignore_(false),
    draw_only_on_pick_(false)
{
  directions_[0]=Vector(1,0,0);
  directions_[1]=Vector(-1,0,0);
  directions_[2]=Vector(0,1,0);
  directions_[3]=Vector(0,-1,0);
  directions_[4]=Vector(0,0,1);
  directions_[5]=Vector(0,0,-1);
}


GeomPick::GeomPick(GeomHandle obj, ModulePickable* module, const Vector& v1)
  : GeomContainer(obj),
    module_(module),
    cbdata_(0),
    picked_obj_(0),
    directions_(2),
    widget_(0),
    selected_(false),
    ignore_(false),
    draw_only_on_pick_(false)
{
  directions_[0]=v1;
  directions_[1]=-v1;
}


GeomPick::GeomPick(GeomHandle obj, ModulePickable* module,
		   const Vector& v1, const Vector& v2)
  : GeomContainer(obj),
    module_(module),
    cbdata_(0),
    picked_obj_(0),
    directions_(4),
    widget_(0),
    selected_(false),
    ignore_(false),
    draw_only_on_pick_(false)
{
  directions_[0]=v1;
  directions_[1]=-v1;
  directions_[2]=v2;
  directions_[3]=-v2;
}


GeomPick::GeomPick(GeomHandle obj, ModulePickable* module,
		   const Vector& v1, const Vector& v2, const Vector& v3)
  : GeomContainer(obj),
    module_(module),
    cbdata_(0),
    picked_obj_(0),
    directions_(6),
    widget_(0),
    selected_(false),
    ignore_(false),
    draw_only_on_pick_(false)
{
  directions_[0]=v1;
  directions_[1]=-v1;
  directions_[2]=v2;
  directions_[3]=-v2;
  directions_[4]=v3;
  directions_[5]=-v3;
}


GeomPick::GeomPick(const GeomPick& copy)
  : GeomContainer(copy),
    module_(copy.module_),
    cbdata_(copy.cbdata_), 
    picked_obj_(copy.picked_obj_),
    directions_(copy.directions_),
    widget_(copy.widget_),
    selected_(copy.selected_),
    ignore_(copy.ignore_),
    highlight_(copy.highlight_),
    draw_only_on_pick_(copy.draw_only_on_pick_)
{
}


GeomObj* GeomPick::clone()
{
  return scinew GeomPick(*this);
}


void
GeomPick::set_highlight(const MaterialHandle& matl)
{
  highlight_ = matl;
}


void
GeomPick::set_module_data(void* cbdata)
{
  cbdata_ = cbdata;
}


void
GeomPick::set_widget_data(int wd)
{
  widget_data_ = wd;
}


void
GeomPick::set_picked_obj(GeomHandle object)
{
  picked_obj_ = object;
}


void
GeomPick::ignore_until_release()
{
  ignore_ = true;
}


void
GeomPick::pick(ViewWindow* viewwindow, const BState& bs )
{
  selected_=true;
  ignore_=false;
  if(widget_)
  {
    widget_->geom_pick(this, viewwindow, widget_data_, bs);
  }
  if(module_)
  {
    module_->geom_pick(this, cbdata_, picked_obj_);
  }
}


void
GeomPick::release(const BState& bs)
{
  selected_=false;
  if(widget_)
  {
    widget_->geom_release(this, widget_data_, bs);
  }
  if(module_)
  {
    module_->geom_release(this, cbdata_, picked_obj_);
  }
}


void
GeomPick::moved(int axis, double distance, const Vector& delta,
		const BState& bs, const Vector &pick_offset)
{
  if(ignore_) { return; }
  if(widget_)
  {
    widget_->geom_moved(this, axis, distance, delta,
			widget_data_, bs, pick_offset);
  }
  if(module_)
  {
    module_->geom_moved(this, axis, distance, delta, cbdata_, picked_obj_);
  }
}


int
GeomPick::nprincipal()
{
  return directions_.size();
}


const Vector &GeomPick::principal(int i)
{
  return directions_[i];
}


void
GeomPick::set_principal(const Vector& v1)
{
  directions_.clear();
  directions_.push_back(v1);
  directions_.push_back(-v1);
}


void
GeomPick::set_principal(const Vector& v1, const Vector& v2)
{
  directions_.clear();
  directions_.push_back(v1);
  directions_.push_back(-v1);
  directions_.push_back(v2);
  directions_.push_back(-v2);
}


void
GeomPick::set_principal(const Vector& v1, const Vector& v2, const Vector& v3)
{
  directions_.clear();
  directions_.push_back(v1);
  directions_.push_back(-v1);
  directions_.push_back(v2);
  directions_.push_back(-v2);
  directions_.push_back(v3);
  directions_.push_back(-v3);
}


#define GEOMPICK_VERSION 1

void
GeomPick::io(Piostream& stream)
{
  stream.begin_class("GeomPick", GEOMPICK_VERSION);
  GeomContainer::io(stream);
  stream.end_class();
}

} // End namespace SCIRun

