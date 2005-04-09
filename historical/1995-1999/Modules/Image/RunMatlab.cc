/*
 *  RunMatlab.cc:  
 *
 *  Written by:
 */

#include <Classlib/Array1.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ColorMapPort.h>
#include <Geom/Grid.h>
#include <Geom/Group.h>
#include <Geom/Line.h>
#include <Geom/Material.h>
#include <Geometry/Point.h>
#include <Math/MinMax.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>
#include "engine.h"
#include <stdio.h>


class RunMatlab : public Module {
   TCLdouble offset;
   TCLdouble scale;

public:
   RunMatlab(const clString& id);
   RunMatlab(const RunMatlab&, int deep);
   virtual ~RunMatlab();
   virtual Module* clone(int deep);
   virtual void execute();

};

extern "C" {
Module* make_RunMatlab(const clString& id)
{
   return scinew RunMatlab(id);
}
};

static clString module_name("RunMatlab");
static clString widget_name("RunMatlab Widget");

RunMatlab::RunMatlab(const clString& id)
: Module("RunMatlab", id, Filter),
  scale("scale", id, this), offset("offset", id, this)
{

}

RunMatlab::RunMatlab(const RunMatlab& copy, int deep)
: Module(copy, deep), 
  scale("scale", id, this), offset("offset", id, this)
{
   NOT_FINISHED("RunMatlab::RunMatlab");
}

RunMatlab::~RunMatlab()
{
}

Module* RunMatlab::clone(int deep)
{
   return scinew RunMatlab(*this, deep);
}

void RunMatlab::execute()
{
    // get the scalar field...if you can

    Engine *ep;

    cerr << "Do we at least get to here??";
    
    if (!(ep = engOpen("/afs/.cs.utah.edu/os/sgi_53/usr/local/apps/dist/matlab/bin/matlab"))) {
      fprintf(stderr, "\nCan't start MATLAB engine\n");
      
    }

    cerr << "Done.. it worked!?!?!";
    
    engClose(ep);

    
/*    ScalarFieldHandle sfield;
    if (!inscalarfield->get( sfield ))
	return;

    ScalarFieldRG* rg=sfield->getRG();
    if(!rg){
      cerr << "RunMatlab cannot handle this field type\n";
      return;
    }
    ScalarFieldRG* newgrid=new ScalarFieldRG();
    newgrid->copy_bounds(rg);
    double s=scale.get();
    double o=offset.get();
    int nx=rg->grid.dim1();
    int ny=rg->grid.dim2();
    int nz=rg->grid.dim3();
    newgrid->resize(nx,ny,nz);
    for(int x=0;x<nx;x++){
      for(int y=0;y<ny;y++){
	for(int z=0;z<nz;z++){
	  newgrid->grid(x,y,z)=rg->grid(x,y,z)*s+o;
	}
      }
    }
             
    outscalarfield->send( newgrid ); */
}


