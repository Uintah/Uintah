//static char *id="@(#) $Id$";

/*
 * This program generates a really simple surface...
 * Peter-Pike Sloan
 */

#include <sci_config.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/BasicSurfaces.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Datatypes/TriSurface.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;
#include <SCICore/Malloc/Allocator.h>
#include <PSECore/Datatypes/GeometryPort.h>

#include <SCICore/Geom/GeomLine.h>

#include <math.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;

class SurfGen : public Module {
    SurfaceOPort* oport;

    TriSurface *ts;

    TCLint nxv;
    TCLint nyv;

    TCLdouble zscalev;
    TCLdouble periodv;
public:
    SurfGen(const clString& id);
    virtual ~SurfGen();
    virtual void execute();
};

extern "C" Module* make_SurfGen(const clString& id) {
  return new SurfGen(id);
}

static clString module_name("SurfGen");

SurfGen::SurfGen(const clString& id)
: Module("SurfGen", id, Filter),ts(0),
  nxv("nx",id,this),nyv("ny",id,this),
  zscalev("zscale",id,this),
  periodv("period",id,this)
{
    // Create the output port
    oport=new SurfaceOPort(this, "Out Surf", SurfaceIPort::Atomic);
    add_oport(oport);

}

SurfGen::~SurfGen()
{
}

extern int placement_policy;

void SurfGen::execute()
{
    SurfaceHandle oSurf;

    cerr << "Init!\n";
    ts = scinew TriSurface();
    
    // ok, just build a "plane" for now...
    
    int nx=nxv.get();
    int ny=nyv.get();
    double zscale = zscalev.get();
    
    double period = periodv.get();

    int y;
    for(y=0;y<ny;y++) {
      double yv = y/(ny-1.0)*M_PI*period;
      double cyv = cos(yv);
      for(int x=0;x<nx;x++) {
	double xv = x/(nx-1.0)*M_PI*period;
	ts->points.add(Point(x,y,cyv*cos(xv)*zscale));
      }
    }
    
    // now add the elements...
    
    for(y=0;y<ny-1;y++) {
      for(int x=0;x<nx-1;x++) {
	int pll = y*nx + x;
	int plr = y*nx + x + 1;
	
	int pul = (y+1)*nx + x;
	int pur = (y+1)*nx + x + 1;
	
	ts->elements.add(new TSElement(pll,plr,pul));
	ts->elements.add(new TSElement(plr,pur,pul));
      }
    }
    
    oSurf = ts;
    
    oport->send(oSurf);
}	

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.8  2000/03/17 09:27:22  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.7  1999/10/07 02:07:00  sparker
// use standard iostreams and complex type
//
// Revision 1.6  1999/08/25 03:48:01  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.5  1999/08/19 23:17:54  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.4  1999/08/19 05:30:53  sparker
// Configuration updates:
//  - renamed config.h to sci_config.h
//  - also uses sci_defs.h, since I couldn't get it to substitute vars in
//    sci_config.h
//  - Added flags for --enable-scirun, --enable-uintah, and
//    --enable-davew, to build the specific package set.  More than one
//    can be specified, and at least one must be present.
//  - Added a --enable-parallel, to build the new parallel version.
//    Doesn't do much yet.
//  - Made construction of config.h a little bit more general
//
// Revision 1.3  1999/08/18 20:19:58  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:43  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:59  mcq
// Initial commit
//
// Revision 1.2  1999/04/29 03:19:28  dav
// updates
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//

