/*
 *  Magnitude.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <PSECommon/Dataflow/Module.h>
#include <PSECommon/Datatypes/ScalarFieldPort.h>
#include <PSECommon/Datatypes/SurfacePort.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <PSECommon/Datatypes/VectorFieldPort.h>
#include <PSECommon/Datatypes/VectorFieldOcean.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Math/Expon.h>
#include <SCICore/Multitask/Task.h>
#include <values.h>

class Magnitude : public Module {
    VectorFieldIPort* infield;
    ScalarFieldOPort* outfield;
public:
    void parallel(int proc);
    VectorFieldOcean* vfield;
    ScalarFieldRG* sfield;
    int np;
    Mutex minmax;
    Magnitude(const clString& id);
    virtual ~Magnitude();
    virtual void execute();
};

Module* make_Magnitude(const clString& id) {
  return new Magnitude(id);
}

Magnitude::Magnitude(const clString& id)
: Module("Magnitude", id, Filter)
{
  infield=new VectorFieldIPort(this, "Vector", VectorFieldIPort::Atomic);
  add_iport(infield);

  // Create the output port
  outfield=new ScalarFieldOPort(this, "Magnitude", ScalarFieldIPort::Atomic);
  add_oport(outfield);
}

Magnitude::Magnitude(const Magnitude& copy, int deep)
: Module(copy, deep)
{
}

Magnitude::~Magnitude()
{
}

static void do_parallel(void* obj, int proc)
{
  Magnitude* module=(Magnitude*)obj;
  module->parallel(proc);
}

  

void Magnitude::parallel(int proc)
{
  int nx=vfield->nx;
  int ny=vfield->ny;
  int nz=vfield->nz;
  int sz=proc*nz/np;		// starting z
  int ez=(proc+1)*nz/np;	// ending z
  double min=MAXDOUBLE;
  double max=-MAXDOUBLE;
  float* p1=vfield->data+nx*ny*sz;
  float* p2=p1+nx*ny*nz;
  for(int k=sz;k<ez;k++){
    if(proc == 0)
      update_progress(k, ez);
    for(int j=0;j<ny;j++){
      for(int i=0;i<nx;i++){
	float u=*p1++;
	float v=*p2++;
	double mag=Sqrt(u*u+v*v);
	sfield->grid(i,j,k)=mag;
	min=Min(min, mag);
	max=Max(max, mag);
      }
    }
  }
  minmax.lock();
  double mn, mx;
  sfield->get_minmax(mn, mx);
  min=Min(mn, min);
  max=Max(mx, max);
  sfield->set_minmax(min, max);
  cerr << proc << ": min=" << min << ", max=" << max << endl;
  minmax.unlock();
}

void Magnitude::execute()
{
  VectorFieldHandle iff;
  if(!infield->get(iff))
    return;
  vfield=iff->getOcean();
  if(!vfield){
    error("Magnitude can't deal with this field");
    return;
  }
  sfield=new ScalarFieldRG();
  sfield->resize(vfield->nx, vfield->ny, vfield->nz);
  Point min, max;
  vfield->get_bounds(min, max);
  sfield->set_bounds(min, max);
  cerr << "setting zgrid, depthval=" << vfield->depthval.size() << endl;
  sfield->zgrid=vfield->depthval;
  cerr << "zgrid=" << sfield->zgrid.size() << endl;
  cerr << "At magnitude: this=" << sfield << ", sizes: " << sfield->xgrid.size() << ", " << sfield->ygrid.size() << ", " << sfield->zgrid.size() << endl;
  sfield->set_minmax(MAXDOUBLE, -MAXDOUBLE);
  np=Task::nprocessors();
  Task::multiprocess(np, do_parallel, this);
  outfield->send(sfield);
}
