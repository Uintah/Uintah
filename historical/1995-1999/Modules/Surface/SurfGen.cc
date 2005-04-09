/*
 * This program generates a really simple surface...
 * Peter-Pike Sloan
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
#include <Datatypes/GeometryPort.h>

#include <Geom/Line.h>

#include <math.h>

class SurfGen : public Module {
    SurfaceOPort* oport;

    TriSurface *ts;

    TCLint nxv;
    TCLint nyv;

    TCLdouble zscalev;
    TCLdouble periodv;
public:
    SurfGen(const clString& id);
    SurfGen(const SurfGen&, int deep);
    virtual ~SurfGen();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_SurfGen(const clString& id)
{
    return new SurfGen(id);
}
};

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

SurfGen::SurfGen(const SurfGen& copy, int deep)
: Module(copy, deep),
  nxv("nx",id,this),nyv("ny",id,this),
  zscalev("zscale",id,this),
  periodv("period",id,this)
{
    NOT_FINISHED("SurfGen::SurfGen");
}

SurfGen::~SurfGen()
{
}

Module* SurfGen::clone(int deep)
{
    return new SurfGen(*this, deep);
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

    for(int y=0;y<ny;y++) {
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


