//static char *id="@(#) $Id$";

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

#include <Util/NotFinished.h>
#include <Dataflow/Module.h>
#include <CommonDatatypes/GeometryPort.h>
#include <CommonDatatypes/ScalarFieldPort.h>
#include <CommonDatatypes/VectorFieldPort.h>
#include <Geom/GeomObj.h>
#include <Geom/GeomLine.h>
#include <Geom/GeomGroup.h>
#include <Geom/Material.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

namespace PSECommon {
namespace Modules {

using namespace PSECommon::Dataflow;
using namespace PSECommon::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

class GenFieldEdges : public Module {
    ScalarFieldIPort* insfield;
    VectorFieldIPort* invfield;
    GeometryOPort* ogeom;
public:
    GenFieldEdges(const clString& id);
    GenFieldEdges(const GenFieldEdges&, int deep);
    virtual ~GenFieldEdges();
    virtual Module* clone(int deep);
    virtual void execute();
    MaterialHandle matl;
};

Module* make_GenFieldEdges(const clString& id) {
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

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:58:13  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:33  dav
// Import sources
//
//
