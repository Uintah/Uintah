//static char *id="@(#) $Id$";

/*
 *  SimpSurface.cc - Simplify a surface using garland/heckbert code
 *
 *  Written by:
 *   Peter-Pike Sloan
 *   Department of Computer Science
 *   University of Utah
 *   July 1998
 *
 *  Copyright (C) 1998 SCI Group
 *
 */

#include <SCICore/config.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Util/NotFinished.h>
#include <PSECommon/Dataflow/Module.h>
#include <SCICore/CoreDatatypes/BasicSurfaces.h>
#include <PSECommon/CommonDatatypes/SurfacePort.h>
#include <SCICore/CoreDatatypes/TriSurface.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream.h>
#include <SCICore/Malloc/Allocator.h>

#include <GHAbstraction.h>

extern int placement_policy;

namespace PSECommon {
namespace Modules {

using PSECommon::Dataflow::Module;
using PSECommon::CommonDatatypes::TriSurface;
using PSECommon::CommonDatatypes::SurfaceIPort;
using PSECommon::CommonDatatypes::SurfaceOPort;
using PSECommon::CommonDatatypes::SurfaceHandle;

using SCICore::TclInterface::TCLint;
using SCICore::Geometry::Point;
using SCICore::Containers::clString;

GHAbstraction::GHAbstraction(TriSurface *surf) {
  InitAdd();

  int i;
  for(i=0;i<surf->points.size();i++) {
    AddPoint(surf->points[i].x(),
	     surf->points[i].y(),
	     surf->points[i].z());
	    
//    M0.in_Vertex(pt);
  }

  for(i=0;i<surf->elements.size();i++) {
    if (surf->elements[i]) {
      AddFace(surf->elements[i]->i1+1,
	      surf->elements[i]->i2+1,
	      surf->elements[i]->i3+1);
    }
  }
  
  FinishAdd();
}
  

void GHAbstraction::DumpSurface(TriSurface* surf)
{
  work = surf;
  RDumpSurface();
}

void GHAbstraction::SAddPoint(double x, double y, double z)
{
  work->add_point(Point(x,y,z));
}

void GHAbstraction::SAddFace(int i, int j, int k)
{
  work->add_triangle(i,j,k);
}


class SimpSurface : public Module {
    SurfaceIPort* iport;
    SurfaceOPort* oport;

    TriSurface *st;
    GHAbstraction *simpObj;
    TCLint numFaces;
    TCLint collapseMode;
public:
    SimpSurface(const clString& id);
    virtual ~SimpSurface();
    virtual void execute();
};

Module* make_SimpSurface(const clString& id) {
   return new SimpSurface(id);
}

static clString module_name("SimpSurface");

SimpSurface::SimpSurface(const clString& id)
: Module("SimpSurface", id, Filter),numFaces("numfaces",id,this),
  collapseMode("collapsemode",id,this),st(0),simpObj(0)
{
    // Create the input ports
    iport=new SurfaceIPort(this, "In Surf", SurfaceIPort::Atomic);
    add_iport(iport);
    // Create the output port
    oport=new SurfaceOPort(this, "Out Surf", SurfaceIPort::Atomic);
    add_oport(oport);
}

SimpSurface::SimpSurface(const SimpSurface& copy, int deep)
: Module(copy, deep),numFaces("numfaces",id,this),
  collapseMode("collapsemode",id,this)
{
    NOT_FINISHED("SimpSurface::SimpSurface");
}

SimpSurface::~SimpSurface()
{
}

void SimpSurface::execute()
{
    SurfaceHandle iSurf;

    if(!iport->get(iSurf))
	return;

    TriSurface *nst = iSurf->getTriSurface();
    if (!nst) return;

    if (nst != st) {
      st = nst;
      
      if (simpObj) delete simpObj;
      simpObj = scinew GHAbstraction(st);
      simpObj->owner = this;
    }

    int nfaces = numFaces.get();

    placement_policy = collapseMode.get();

    simpObj->Simplify(nfaces);

    TriSurface *newSurf = scinew TriSurface;

    simpObj->DumpSurface(newSurf);
    
    SurfaceHandle oSurf(newSurf);

    oport->send(oSurf);
}	

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
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
// Revision 1.1  1999/07/27 16:57:58  mcq
// Initial commit
//
// Revision 1.2  1999/04/29 03:19:28  dav
// updates
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
