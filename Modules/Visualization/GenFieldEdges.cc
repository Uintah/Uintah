
/*
 *  GenFieldEdges.cc:  IsoSurfaces a SFRG bitwise
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <config.h>
#undef SCI_ASSERTION_LEVEL_3
#define SCI_ASSERTION_LEVEL_2
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Geom/Geom.h>
#include <Geom/Line.h>
#include <Geom/Group.h>
#include <iostream.h>

class GenFieldEdges : public Module {
    ScalarFieldIPort* infield;
    GeometryOPort* ogeom;
    int geom_id;
public:
    GenFieldEdges(const clString& id);
    GenFieldEdges(const GenFieldEdges&, int deep);
    virtual ~GenFieldEdges();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_GenFieldEdges(const clString& id)
{
    return new GenFieldEdges(id);
}
};

GenFieldEdges::GenFieldEdges(const clString& id)
: Module("GenFieldEdges", id, Filter)
{
    // Create the input ports
    infield=new ScalarFieldIPort(this, "Field", ScalarFieldIPort::Atomic);
    add_iport(infield);
    // Create the output port
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
    geom_id=0;
}

GenFieldEdges::GenFieldEdges(const GenFieldEdges& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("GenFieldEdges::GenFieldEdges");
}

GenFieldEdges::~GenFieldEdges()
{
}

Module* GenFieldEdges::clone(int deep)
{
    return new GenFieldEdges(*this, deep);
}

void GenFieldEdges::execute()
{
    ScalarFieldHandle field;
    if(!infield->get(field))
	return;
    ScalarFieldRG* regular_grid=field->getRG();
    if(!regular_grid){
	error("GenFieldEdges can't handle irregular grids");
	return;
    }
    Point p1, p2;
    regular_grid->get_bounds(p1, p2);
    if (geom_id) {
	ogeom->delObj(geom_id);
    }
    GeomGroup *g = new GeomGroup;
    GeomObj *l1, *l2, *l3, *l4, *l5, *l6, *l7, *l8, *l9, *l10, *l11, *l12;
    l1 =new GeomLine(Point(p1.x(),p1.y(),p1.z()), Point(p2.x(),p1.y(),p1.z()));
    l2 =new GeomLine(Point(p2.x(),p1.y(),p1.z()), Point(p2.x(),p2.y(),p1.z()));
    l3 =new GeomLine(Point(p2.x(),p2.y(),p1.z()), Point(p1.x(),p2.y(),p1.z()));
    l4 =new GeomLine(Point(p1.x(),p2.y(),p1.z()), Point(p1.x(),p1.y(),p1.z()));
    l5 =new GeomLine(Point(p1.x(),p1.y(),p2.z()), Point(p2.x(),p1.y(),p2.z()));
    l6 =new GeomLine(Point(p2.x(),p1.y(),p2.z()), Point(p2.x(),p2.y(),p2.z()));
    l7 =new GeomLine(Point(p2.x(),p2.y(),p2.z()), Point(p1.x(),p2.y(),p2.z()));
    l8 =new GeomLine(Point(p1.x(),p2.y(),p2.z()), Point(p1.x(),p1.y(),p2.z()));
    l9 =new GeomLine(Point(p1.x(),p1.y(),p1.z()), Point(p1.x(),p1.y(),p2.z()));
    l10=new GeomLine(Point(p1.x(),p2.y(),p1.z()), Point(p1.x(),p2.y(),p2.z()));
    l11=new GeomLine(Point(p2.x(),p1.y(),p1.z()), Point(p2.x(),p1.y(),p2.z()));
    l12=new GeomLine(Point(p2.x(),p2.y(),p1.z()), Point(p2.x(),p2.y(),p2.z()));
    g->add(l1);
    g->add(l2);
    g->add(l3);
    g->add(l4);
    g->add(l5);
    g->add(l6);
    g->add(l7);
    g->add(l8);
    g->add(l9);
    g->add(l10);
    g->add(l11);
    g->add(l12);
    geom_id=ogeom->addObj(g, "Bounding Box");
    ogeom->flushViews();
}
