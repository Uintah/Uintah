
/*
 *  Pick.h: Picking information for Geometry objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Pick.h>
#include <Dataflow/Module.h>
#include <Widgets/BaseWidget.h>

GeomPick::GeomPick(GeomObj* obj, Module* module)
: GeomContainer(obj), module(module), cbdata(0),
  directions(6), widget(0), selected(0)
{
    directions[0]=Vector(1,0,0);
    directions[1]=Vector(-1,0,0);
    directions[2]=Vector(0,1,0);
    directions[3]=Vector(0,-1,0);
    directions[4]=Vector(0,0,1);
    directions[5]=Vector(0,0,-1);
}

GeomPick::GeomPick(GeomObj* obj, Module* module,
		   BaseWidget* widget, int widget_data)
: GeomContainer(obj), module(module), cbdata(0),
  directions(6), widget(widget), widget_data(widget_data), selected(0)
{
    directions[0]=Vector(1,0,0);
    directions[1]=Vector(-1,0,0);
    directions[2]=Vector(0,1,0);
    directions[3]=Vector(0,-1,0);
    directions[4]=Vector(0,0,1);
    directions[5]=Vector(0,0,-1);
}

GeomPick::GeomPick(GeomObj* obj, Module* module, const Vector& v1)
: GeomContainer(obj), module(module), directions(2), cbdata(0),
  widget(0), selected(0)
{
    directions[0]=v1;
    directions[1]=-v1;
}

GeomPick::GeomPick(GeomObj* obj, Module* module, const Vector& v1, const Vector& v2)
: GeomContainer(obj), module(module), directions(4), cbdata(0),
  widget(0), selected(0)
{
    directions[0]=v1;
    directions[1]=-v1;
    directions[2]=v2;
    directions[3]=-v2;
}

GeomPick::GeomPick(GeomObj* obj, Module* module, const Vector& v1, const Vector& v2,
		   const Vector& v3)
: GeomContainer(obj), module(module), directions(6), cbdata(0),
  widget(0), selected(0)
{
    directions[0]=v1;
    directions[1]=-v1;
    directions[2]=v2;
    directions[3]=-v2;
    directions[4]=v3;
    directions[5]=-v3;
}

GeomPick::GeomPick(const GeomPick& copy)
: GeomContainer(copy), directions(copy.directions), highlight(copy.highlight),
  cbdata(copy.cbdata), module(copy.module), widget(copy.widget),
  selected(copy.selected)
{
}

GeomObj* GeomPick::clone()
{
    return new GeomPick(*this);
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

void GeomPick::pick()
{
    selected=1;
    if(widget)
	widget->geom_pick(widget_data);
    if(module)
	module->geom_pick(cbdata);
}

void GeomPick::release()
{
    selected=0;
    if(widget)
	widget->geom_release(widget_data);
    if(module)
	module->geom_release(cbdata);
}

void GeomPick::moved(int axis, double distance, const Vector& delta)
{
    if(widget)
	widget->geom_moved(axis, distance, delta, widget_data);
    if(module)
	module->geom_moved(axis, distance, delta, cbdata);
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
