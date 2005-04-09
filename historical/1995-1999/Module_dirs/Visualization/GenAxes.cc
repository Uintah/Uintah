/*
 *  GenAxes.cc:  
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Mar. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/GeometryPort.h>
#include <Geom/Cylinder.h>
#include <Geom/Cone.h>
#include <Geom/Group.h>
#include <Geom/Material.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>

class GenAxes : public Module {
    TCLdouble size;
    int size_changed;
    GeometryOPort* ogeom;
    MaterialHandle dk_red;
    MaterialHandle dk_green;
    MaterialHandle dk_blue;
    MaterialHandle lt_red;
    MaterialHandle lt_green;
    MaterialHandle lt_blue;
public:
    GenAxes(const clString& id);
    GenAxes(const GenAxes&, int deep);
    virtual ~GenAxes();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void tcl_command(TCLArgs&, void*);
};

static Module* make_GenAxes(const clString& id)
{
   return scinew GenAxes(id);
}

static RegisterModule db1("Visualization", "GenAxes", make_GenAxes);

static clString module_name("GenAxes");

GenAxes::GenAxes(const clString& id)
: Module("GenAxes", id, Source), size_changed(1), size("size", id, this)
{
   // Create the output port
   ogeom = scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
   add_oport(ogeom);
   dk_red = scinew Material(Color(0,0,0), Color(.2,0,0),
			 Color(.5,.5,.5), 20);
   dk_green = scinew Material(Color(0,0,0), Color(0,.2,0),
			   Color(.5,.5,.5), 20);
   dk_blue = scinew Material(Color(0,0,0), Color(0,0,.2),
			  Color(.5,.5,.5), 20);
   lt_red = scinew Material(Color(0,0,0), Color(.8,0,0),
			 Color(.5,.5,.5), 20);
   lt_green = scinew Material(Color(0,0,0), Color(0,.8,0),
			   Color(.5,.5,.5), 20);
   lt_blue = scinew Material(Color(0,0,0), Color(0,0,.8),
			  Color(.5,.5,.5), 20);
}

GenAxes::GenAxes(const GenAxes& copy, int deep)
: Module(copy, deep), size_changed(1), size("size", id, this)
{
   NOT_FINISHED("GenAxes::GenAxes");
}

GenAxes::~GenAxes()
{
}

Module* GenAxes::clone(int deep)
{
   return scinew GenAxes(*this, deep);
}

void GenAxes::execute()
{
   if (!size_changed) return;
   size_changed=0;
   ogeom->delAll();
   GeomGroup* xp = scinew GeomGroup;
   GeomGroup* yp = scinew GeomGroup;
   GeomGroup* zp = scinew GeomGroup;
   GeomGroup* xn = scinew GeomGroup;
   GeomGroup* yn = scinew GeomGroup;
   GeomGroup* zn = scinew GeomGroup;
   double sz=size.get();
//   cerr << "Size= " <<size.get() << "\n";
   xp->add(scinew GeomCylinder(Point(0,0,0), Point(sz, 0, 0), sz/20));
   xp->add(scinew GeomCone(Point(sz, 0, 0), Point(sz+sz/5, 0, 0), sz/10, 0));
   yp->add(scinew GeomCylinder(Point(0,0,0), Point(0, sz, 0), sz/20));
   yp->add(scinew GeomCone(Point(0, sz, 0), Point(0, sz+sz/5, 0), sz/10, 0));
   zp->add(scinew GeomCylinder(Point(0,0,0), Point(0, 0, sz), sz/20));
   zp->add(scinew GeomCone(Point(0, 0, sz), Point(0, 0, sz+sz/5), sz/10, 0));
   xn->add(scinew GeomCylinder(Point(0,0,0), Point(-sz, 0, 0), sz/20));
   xn->add(scinew GeomCone(Point(-sz, 0, 0), Point(-sz-sz/5, 0, 0), sz/10, 0));
   yn->add(scinew GeomCylinder(Point(0,0,0), Point(0, -sz, 0), sz/20));
   yn->add(scinew GeomCone(Point(0, -sz, 0), Point(0, -sz-sz/5, 0), sz/10, 0));
   zn->add(scinew GeomCylinder(Point(0,0,0), Point(0, 0, -sz), sz/20));
   zn->add(scinew GeomCone(Point(0, 0, -sz), Point(0, 0, -sz-sz/5), sz/10, 0));
   GeomGroup* all=scinew GeomGroup;
   all->add(scinew GeomMaterial(xp, lt_red));
   all->add(scinew GeomMaterial(yp, lt_green));
   all->add(scinew GeomMaterial(zp, lt_blue));
   all->add(scinew GeomMaterial(xn, dk_red));
   all->add(scinew GeomMaterial(yn, dk_green));
   all->add(scinew GeomMaterial(zn, dk_blue));
   ogeom->addObj(all, "Axes");
   ogeom->flushViews();
}

void GenAxes::tcl_command(TCLArgs& args, void* userdata)
{
    if(args.count() < 2){
	args.error("GenAxes needs a minor command");
	return;
    }
    if (args[1] == "size_changed") {
	size_changed=1;
	want_to_execute();
    } else {
	Module::tcl_command(args, userdata);
    }
}
