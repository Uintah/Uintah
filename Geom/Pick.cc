
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
#include <Geom/PickMessage.h>
#include <Dataflow/Module.h>

GeomPick::GeomPick(Module* module)
: module(module), mailbox(0), cbdata(0)
{
}

GeomPick::GeomPick(Module* module, const Vector& v1)
: module(module), directions(2), mailbox(0), cbdata(0)
{
    directions[0]=v1;
    directions[1]=-v1;
}

GeomPick::GeomPick(Module* module, const Vector& v1, const Vector& v2)
: module(module), directions(4), mailbox(0), cbdata(0)
{
    directions[0]=v1;
    directions[1]=-v1;
    directions[2]=v2;
    directions[3]=-v2;
}

GeomPick::GeomPick(Module* module, const Vector& v1, const Vector& v2,
		   const Vector& v3)
: module(module), directions(6), mailbox(0), cbdata(0)
{
    directions[0]=v1;
    directions[1]=-v1;
    directions[2]=v2;
    directions[3]=-v2;
    directions[4]=v3;
    directions[5]=-v3;
}

GeomPick::~GeomPick()
{
}

void GeomPick::set_highlight(const MaterialHandle& matl)
{
    hightlight=matl;
}

void GeomPick::set_cbdata(void* _cbdata)
{
    cbdata=_cbdata;
}

void GeomPick::pick()
{
    if(mailbox){
	// Send a message...
        mailbox->send(new GeomPickMessage(module, cbdata));
    } else {
	// Do it directly..
	module->geom_pick(cbdata);
    }
}

void GeomPick::release()
{
    if(mailbox){
	// Send a message...
        mailbox->send(new GeomPickMessage(module, cbdata, 0));
    } else {
	// Do it directly..
	module->geom_release(cbdata);
    }
}

void GeomPick::moved(int axis, double distance, const Vector& delta)
{
    if(mailbox){
	// Send a message...
        mailbox->send(new GeomPickMessage(module, axis, distance, delta, cbdata));
    } else {
	cerr << "Calling moved...\n";
	module->geom_moved(axis, distance, delta, cbdata);
	cerr << "done\n";
    }
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
