//static char *id="@(#) $Id$";

/*
 *  FieldCage.cc:  IsoSurfaces a SFRG bitwise
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
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using namespace SCICore::Math;

class FieldCage : public Module {
    ScalarFieldIPort* insfield;
    VectorFieldIPort* invfield;
    GeometryOPort* ogeom;
    MaterialHandle dk_red;
    MaterialHandle dk_green;
    MaterialHandle dk_blue;
    MaterialHandle lt_red;
    MaterialHandle lt_green;
    MaterialHandle lt_blue;
    MaterialHandle gray;
    TCLint numx;
    TCLint numy;
    TCLint numz;
public:
    FieldCage(const clString& id);
    virtual ~FieldCage();
    virtual void execute();
    MaterialHandle matl;
};

Module* make_FieldCage(const clString& id) {
  return new FieldCage(id);
}

FieldCage::FieldCage(const clString& id)
: Module("FieldCage", id, Filter), numx("numx", id, this), numy("numy", id, this), numz("numz", id, this)
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
   dk_red = scinew Material(Color(0,0,0), Color(.3,0,0),
			 Color(.5,.5,.5), 20);
   dk_green = scinew Material(Color(0,0,0), Color(0,.3,0),
			   Color(.5,.5,.5), 20);
   dk_blue = scinew Material(Color(0,0,0), Color(0,0,.3),
			  Color(.5,.5,.5), 20);
   lt_red = scinew Material(Color(0,0,0), Color(.8,0,0),
			 Color(.5,.5,.5), 20);
   lt_green = scinew Material(Color(0,0,0), Color(0,.8,0),
			   Color(.5,.5,.5), 20);
   lt_blue = scinew Material(Color(0,0,0), Color(0,0,.8),
			  Color(.5,.5,.5), 20);
    gray = scinew Material(Color(0,0,0), Color(.4,.4,.4),
			  Color(.5,.5,.5), 20);
}

FieldCage::~FieldCage()
{
}

void FieldCage::execute()
{
    ogeom->delAll();

    ScalarFieldHandle sfield;
    Point min, max;
    bool haveit=false;
    if(insfield->get(sfield)){
	sfield->get_bounds(min, max);
	haveit=true;
    }
    VectorFieldHandle vfield;
    if(invfield->get(vfield)){
	vfield->get_bounds(min, max);
	haveit=true;
    }
    if(!haveit)
	return;
    GeomGroup* all=new GeomGroup();
    GeomLines* xmin=new GeomLines();
    GeomLines* ymin=new GeomLines();
    GeomLines* zmin=new GeomLines();
    GeomLines* xmax=new GeomLines();
    GeomLines* ymax=new GeomLines();
    GeomLines* zmax=new GeomLines();
    GeomLines* edges=new GeomLines();
    all->add(new GeomMaterial(xmin, dk_red));
    all->add(new GeomMaterial(ymin, dk_green));
    all->add(new GeomMaterial(zmin, dk_blue));
    all->add(new GeomMaterial(xmax, lt_red));
    all->add(new GeomMaterial(ymax, lt_green));
    all->add(new GeomMaterial(zmax, lt_blue));
    all->add(new GeomMaterial(edges, gray));

    edges->add(Point(min.x(), min.y(), min.z()), Point(min.x(), min.y(), max.z()));
    edges->add(Point(min.x(), min.y(), min.z()), Point(min.x(), max.y(), min.z()));
    edges->add(Point(min.x(), min.y(), min.z()), Point(max.x(), min.y(), min.z()));
    edges->add(Point(max.x(), min.y(), min.z()), Point(max.x(), max.y(), min.z()));
    edges->add(Point(max.x(), min.y(), min.z()), Point(max.x(), min.y(), max.z()));
    edges->add(Point(min.x(), max.y(), min.z()), Point(max.x(), max.y(), min.z()));
    edges->add(Point(min.x(), max.y(), min.z()), Point(min.x(), max.y(), max.z()));
    edges->add(Point(min.x(), min.y(), min.z()), Point(min.x(), min.y(), max.z()));
    edges->add(Point(min.x(), min.y(), max.z()), Point(max.x(), min.y(), max.z()));
    edges->add(Point(min.x(), min.y(), max.z()), Point(min.x(), max.y(), max.z()));
    edges->add(Point(max.x(), max.y(), min.z()), Point(max.x(), max.y(), max.z()));
    edges->add(Point(max.x(), min.y(), max.z()), Point(max.x(), max.y(), max.z()));
    edges->add(Point(min.x(), max.y(), max.z()), Point(max.x(), max.y(), max.z()));
    int i;
    int nx=numx.get();
    int ny=numy.get();
    int nz=numz.get();
    for(i=0;i<nx;i++){
	double x=Interpolate(min.x(), max.x(), double(i+1)/double(nx+1));
	ymin->add(Point(x, min.y(), min.z()), Point(x, min.y(), max.z()));
	ymax->add(Point(x, max.y(), min.z()), Point(x, max.y(), max.z()));
	zmin->add(Point(x, min.y(), min.z()), Point(x, max.y(), min.z()));
	zmax->add(Point(x, min.y(), max.z()), Point(x, max.y(), max.z()));
    }
    for(i=0;i<ny;i++){
	double y=Interpolate(min.y(), max.y(), double(i+1)/double(ny+1));
	xmin->add(Point(min.x(), y, min.z()), Point(min.x(), y, max.z()));
	xmax->add(Point(max.x(), y, min.z()), Point(max.x(), y, max.z()));
	zmin->add(Point(min.x(), y, min.z()), Point(max.x(), y, min.z()));
	zmax->add(Point(min.x(), y, max.z()), Point(max.x(), y, max.z()));
    }
    for(i=0;i<nz;i++){
	double z=Interpolate(min.z(), max.z(), double(i+1)/double(nz+1));
	xmin->add(Point(min.x(), min.y(), z), Point(min.x(), max.y(), z));
	xmax->add(Point(max.x(), min.y(), z), Point(max.x(), max.y(), z));
	ymin->add(Point(min.x(), min.y(), z), Point(max.x(), min.y(), z));
	ymax->add(Point(min.x(), max.y(), z), Point(max.x(), max.y(), z));
    }
    ogeom->delAll();
    ogeom->addObj(all, "Field Cage");
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.5  1999/08/25 03:48:06  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:56  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:20:05  sparker
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
// Revision 1.1  1999/07/27 16:58:12  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:34  dav
// Import sources
//
//
