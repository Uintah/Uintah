
/*
 *  mySimpSurface - Simplify a surface using garland/heckbert code
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

#include <config.h>
#include <Classlib/String.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/BasicSurfaces.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/TriSurface.h>
#include <TCL/TCLvar.h>
#include <iostream.h>
#include <Malloc/Allocator.h>
#include <Modules/Simplify/SimpObj2d.h>

class mySimpSurface : public Module {
    SurfaceIPort* iport;
    SurfaceOPort* oport;

    TriSurface *st;
    
    SimpObj2d *simpObj;

    TCLint numFaces;
    TCLint collapseMode;
public:
    mySimpSurface(const clString& id);
    mySimpSurface(const mySimpSurface&, int deep);
    virtual ~mySimpSurface();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_mySimpSurface(const clString& id)
{
    return new mySimpSurface(id);
}
};

static clString module_name("mySimpSurface");

mySimpSurface::mySimpSurface(const clString& id)
: Module("mySimpSurface", id, Filter),numFaces("numfaces",id,this),
  collapseMode("collapsemode",id,this),st(0),simpObj(0)
{
    // Create the input ports
    iport=new SurfaceIPort(this, "In Surf", SurfaceIPort::Atomic);
    add_iport(iport);
    // Create the output port
    oport=new SurfaceOPort(this, "Out Surf", SurfaceIPort::Atomic);
    add_oport(oport);
}

mySimpSurface::mySimpSurface(const mySimpSurface& copy, int deep)
: Module(copy, deep),numFaces("numfaces",id,this),
  collapseMode("collapsemode",id,this)
{
    NOT_FINISHED("mySimpSurface::mySimpSurface");
}

mySimpSurface::~mySimpSurface()
{
}

Module* mySimpSurface::clone(int deep)
{
    return new mySimpSurface(*this, deep);
}

extern int placement_policy;

void mySimpSurface::execute()
{
    SurfaceHandle iSurf;

    if(!iport->get(iSurf))
	return;

    TriSurface *nst = iSurf->getTriSurface();
    if (!nst) return;

    if (nst != st) {
      cerr << "Init!\n";
      st = nst;
      
      if (simpObj) delete simpObj;
      simpObj = scinew SimpObj2d();
      //simpObj->owner = this;
      simpObj->Init(nst);
#if 1
      simpObj->ComputeQuadrics();

      simpObj->ErrFunc = new QuadricError2d(simpObj);
      simpObj->FillQ();
#endif

    }
    cerr << "Simplifying!\n";

    static int toggle=1;
#if 1
    int nfaces = numFaces.get();

    simpObj->PopQE(nfaces);// just do pops...

    simpObj->mesh2d.Validate();

    TriSurface *newSurf = scinew TriSurface;

    simpObj->DumpSurface(newSurf);
    
    SurfaceHandle oSurf(newSurf);

    oport->send(oSurf);
#endif
#if 0
    Array1<int> tmpList(0,200);
    int sum=0;

    if (toggle) {
      
      for(int j=0;j<100;j++) {

      for(int i=0;i<simpObj->mesh2d.points.size();i++) {
	tmpList.resize(0);
	simpObj->mesh2d.GetFaceRing(i,tmpList);
	sum += tmpList.size();
#if 0
	cerr << i << ": ";

	for(int j=0;j<tmpList.size();j++) {
	  cerr << tmpList[j] << " ";
	}
	cerr << endl;
#endif
      }
    }
      toggle = 0;
    } else {
      for(int j=0;j<100;j++) {	
      for(int i=0;i<simpObj->mesh2d.points.size();i++) {
	tmpList.resize(0);
	simpObj->mesh2d.GetFaceRing2(i,tmpList);
	sum += tmpList.size();
#if 0
	cerr << i << ": ";

	for(int j=0;j<tmpList.size();j++) {
	  cerr << tmpList[j] << " ";
	}
	cerr << endl;
#endif
      }
    }
      toggle = 1;
    }

    cerr << toggle << " " << sum << endl;
#endif
}	
