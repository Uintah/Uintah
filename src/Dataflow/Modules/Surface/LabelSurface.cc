//static char *id="@(#) $Id$";

/*
 *  LabelSurface.cc:  Label a specific surf in a surftree
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1997
 *
 *  Copyright (C) 1997 SCI Group
 *
 */

#include <sci_config.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/BasicSurfaces.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Datatypes/SurfTree.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Geometry/BBox.h>

namespace PSECommon {
namespace Modules {

using PSECore::Dataflow::Module;
using PSECore::Datatypes::SurfTree;
using PSECore::Datatypes::SurfaceIPort;
using PSECore::Datatypes::SurfaceOPort;
using PSECore::Datatypes::SurfaceHandle;
using PSECore::Datatypes::TriSurface;

using namespace SCICore::TclInterface;
using SCICore::Containers::Array1;

class LabelSurface : public Module {
    SurfaceIPort* iport;
    SurfaceOPort* oport;
    TCLint numberf;
    TCLstring namef;
    int generation;
    Array1<clString> origNames;
public:
    LabelSurface(const clString& id);
    virtual ~LabelSurface();
    virtual void execute();
};

Module* make_LabelSurface(const clString& id) {
  return new LabelSurface(id);
}

//static clString module_name("LabelSurface");

LabelSurface::LabelSurface(const clString& id)
: Module("LabelSurface", id, Filter), generation(-1),
  numberf("numberf", id, this), namef("namef", id, this)
{
    // Create the input ports
    iport=new SurfaceIPort(this, "In Surf", SurfaceIPort::Atomic);
    add_iport(iport);
    // Create the output port
    oport=new SurfaceOPort(this, "Out Surf", SurfaceIPort::Atomic);
    add_oport(oport);
}

LabelSurface::~LabelSurface()
{
}

void LabelSurface::execute()
{
    SurfaceHandle iSurf;
    int i;
    if(!iport->get(iSurf))
	return;
    SurfTree* st=iSurf->getSurfTree();
    TriSurface *ts=iSurf->getTriSurface();
    if (st) {
      if (st->generation != generation) {
	  origNames.resize(0);
	  for (i=0; i<st->surfI.size(); i++) origNames.add(st->surfI[i].name);
      }
      int fnum=numberf.get();
      clString fname=namef.get();
      
      for (i=0; i<st->surfI.size(); i++)
	  st->surfI[i].name=origNames[i];
      
      if (fnum>0 && fnum<st->surfI.size() && 
	  st->surfI[fnum].name != fname) {
	cerr << "Added label: "<<fname<<" to surface number: "<<fnum<<"\n";
	st->surfI[fnum].name=fname;
      }
    } else if (ts) {
      ts->name = namef.get();
    }
    oport->send(iSurf);
}	

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.6  1999/08/25 03:48:00  sparker
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
// Revision 1.3  1999/08/18 20:19:56  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:42  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:57  mcq
// Initial commit
//
// Revision 1.2  1999/04/29 03:19:27  dav
// updates
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
