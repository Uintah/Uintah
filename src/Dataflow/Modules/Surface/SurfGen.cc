
/*
 * This program generates a really simple surface...
 * Peter-Pike Sloan
 */

#include <sci_config.h>
#include <Core/Containers/String.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/BasicSurfaces.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Core/Datatypes/TriSurface.h>
#include <Core/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Ports/GeometryPort.h>

#include <Core/Geom/GeomLine.h>

#include <math.h>

namespace SCIRun {


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

} // End namespace SCIRun


