
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

#include <Core/config.h>
#include <Core/Containers/String.h>
#include <Core/Util/NotFinished.h>
#include <Dataflow/Dataflow/Module.h>
#include <Core/Datatypes/BasicSurfaces.h>
#include <Dataflow/Datatypes/SurfacePort.h>
#include <Core/Datatypes/TriSurface.h>
#include <Core/TclInterface/TCLvar.h>
#include <Core/Malloc/Allocator.h>

#include <GHAbstraction.h>

extern int placement_policy;

namespace SCIRun {



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

extern "C" Module* make_SimpSurface(const clString& id) {
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

} // End namespace SCIRun

