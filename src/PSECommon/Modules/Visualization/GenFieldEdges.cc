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

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/VectorFieldPort.h>
#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Malloc/Allocator.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

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

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.6  1999/09/08 02:26:36  sparker
// Various #include cleanups
//
// Revision 1.5  1999/08/25 03:48:06  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:57  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:20:06  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:48  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:13  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:33  dav
// Import sources
//
//
