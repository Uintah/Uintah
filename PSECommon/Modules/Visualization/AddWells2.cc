//static char *id="@(#) $Id$";

/*
 *  AddWells2.cc:  
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Mar. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/CommonDatatypes/GeometryPort.h>
#include <SCICore/Geom/GeomCylinder.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <fstream.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

class AddWells2 : public Module {
    TCLstring filename;
    TCLdouble radius;
    GeometryOPort* ogeom;
    int ndeep;
    double* dp;
    double* ms;
    double toms(double depth);
public:
    AddWells2(const clString& id);
    virtual ~AddWells2();
    virtual void execute();
};

Module* make_AddWells2(const clString& id) {
 return new AddWells2(id);
}

//static clString module_name("AddWells2");

AddWells2::AddWells2(const clString& id)
: Module("AddWells2", id, Source),
  filename("filename", id, this), radius("radius", id, this)
{
   // Create the output port
   ogeom = scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
   add_oport(ogeom);
}

AddWells2::~AddWells2()
{
}

void AddWells2::execute()
{
    clString fn(filename.get());
    ifstream in(fn());
    if(!in){
	cerr << "Didn't open file: " << fn << '\n';
	return;
    }
    double rad=radius.get();
    GeomGroup* all=new GeomGroup;
    GeomGroup* h[10];
    for(int i=0;i<9;i++){
	h[i]=new GeomGroup();
	HSVColor color(i*360/9., 0.7, 1.0);
	MaterialHandle matl=new Material(color*.2, color*.8, Color(.6,.6,.6),20);
	all->add(new GeomMaterial(h[i], matl));
    }
    char line[1000];
    in.getline(line, 1000);
    while(in){
	double x, y;
	double dummy;
	in >> dummy >> x >> y >> dummy >> dummy >> dummy;
	double z1;
	in >> dummy >> dummy >> z1;
	if(!in)break;
	for(int i=0;i<9;i++){
	    double z2;
	    in >> dummy >> dummy >> z2;
	    if(!in)break;
	    if(z2 > 50000)
		continue;
	    GeomCylinder* cyl=new GeomCylinder(Point(x,y,z1),
					       Point(x,y,z2),
					       rad, 12, 2);
	    h[i]->add(cyl);
	    z1=z2;
	}
	if(!in)break;
    }
    ogeom->delAll();
    ogeom->addObj(all, "Wells");
    ogeom->flushViews();
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.4  1999/08/19 23:17:55  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:20:01  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:46  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:10  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:33  dav
// Import sources
//
//
