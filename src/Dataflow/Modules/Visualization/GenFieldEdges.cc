
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

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Dataflow/Ports/VectorFieldPort.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


class GenFieldEdges : public Module {
    ScalarFieldIPort* insfield;
    VectorFieldIPort* invfield;
    GeometryOPort* ogeom;
public:
    GenFieldEdges(const clString& id);
    virtual ~GenFieldEdges();
    virtual void execute();
    MaterialHandle matl;
};

extern "C" Module* make_GenFieldEdges(const clString& id) {
  return new GenFieldEdges(id);
}

GenFieldEdges::GenFieldEdges(const clString& id)
: Module("GenFieldEdges", id, Filter)
{
    // Create the input ports
    insfield=new ScalarFieldIPort(this, "Scalar Field", ScalarFieldIPort::Atomic);
    add_iport(insfield);
    invfield=new VectorFieldIPort(this, "Vector Field", VectorFieldIPort::Atomic);
    add_iport(invfield);
    // Create the output port
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
    matl=scinew Material(Color(0,0,0), Color(.8,.8,.8),
			 Color(.7,.7,.7), 50);
}

GenFieldEdges::~GenFieldEdges()
{
}

void GenFieldEdges::execute()
{
    ogeom->delAll();

    ScalarFieldHandle sfield;
    if(insfield->get(sfield)){
	Array1<Point> pts;
	sfield->get_boundary_lines(pts);
	GeomLines* lines=new GeomLines();
	for(int i=0;i<pts.size();i+=2)
	    lines->add(pts[i], pts[i+1]);

	ogeom->addObj(new GeomMaterial(lines, matl), "Field Boundary");
    }
    VectorFieldHandle vfield;
    if(invfield->get(vfield)){
	Array1<Point> pts;
	vfield->get_boundary_lines(pts);
	GeomLines* lines=new GeomLines();
	for(int i=0;i<pts.size();i+=2)
	    lines->add(pts[i], pts[i+1]);

	ogeom->addObj(new GeomMaterial(lines, matl), "Field Boundary");
    }
}

} // End namespace SCIRun

