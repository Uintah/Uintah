/*
 *  Downsample.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1996
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Geometry/Point.h>
#include <TCL/TCLvar.h>
#include <Math/Expon.h>
#include <Multitask/Task.h>
#include <values.h>

class Downsample : public Module {
    ScalarFieldIPort* infield;
    ScalarFieldOPort* outfield;
public:
    void parallel(int proc);
    ScalarFieldRG* ifield;
    ScalarFieldRG* ofield;
    int np;
    TCLint downsamplex, downsampley, downsamplez;
    int dsx, dsy, dsz;
    Downsample(const clString& id);
    Downsample(const Downsample&, int deep);
    virtual ~Downsample();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_Downsample(const clString& id)
{
    return new Downsample(id);
}
};

Downsample::Downsample(const clString& id)
: Module("Downsample", id, Filter), downsamplex("downsamplex", id, this),
  downsampley("downsampley", id, this), downsamplez("downsamplez", id, this)
{
  infield=new ScalarFieldIPort(this, "Input field", ScalarFieldIPort::Atomic);
  add_iport(infield);

  // Create the output port
  outfield=new ScalarFieldOPort(this, "Downsampled field", ScalarFieldIPort::Atomic);
  add_oport(outfield);
}

Downsample::Downsample(const Downsample& copy, int deep)
: Module(copy, deep), downsamplex("downsamplex", id, this),
  downsampley("downsampley", id, this), downsamplez("downsamplez", id, this)
{
}

Downsample::~Downsample()
{
}

Module* Downsample::clone(int deep)
{
    return new Downsample(*this, deep);
}

static void do_parallel(void* obj, int proc)
{
  Downsample* module=(Downsample*)obj;
  module->parallel(proc);
}

void Downsample::parallel(int proc)
{
  int nx=ofield->nx;
  int ny=ofield->ny;
  int nz=ofield->nz;
  int sx=proc*nx/np;
  int ex=(proc+1)*nx/np;
  for(int i=sx;i<ex;i++){
    if(proc == 0)
      update_progress(i, ex);
    for(int j=0;j<ny;j++){
      for(int k=0;k<nz;k++){
	int si=i*dsx;
	int ei=Min(si+dsx, ifield->nx);
	int sj=j*dsy;
	int ej=Min(sj+dsy, ifield->ny);
	int sk=k*dsz;
	int ek=Min(sk+dsz, ifield->nz);
	double s=0;
	for(int ii=si;ii<ei;ii++){
	  for(int jj=sj;jj<ej;jj++){
	    for(int kk=sk;kk<ek;kk++){
	      s+=ifield->grid(ii, jj, kk);
	    }
	  }
	}
	s/=((ei-si)*(ej-sj)*(ek-sk));
	ofield->grid(i,j,k)=s;
      }
    }
  }
}

void Downsample::execute()
{
  ScalarFieldHandle iff;
  if(!infield->get(iff))
    return;
  ifield=iff->getRG();
  if(!ifield){
    error("Downsample can't deal with this field");
    return;
  }
  dsx=downsamplex.get();
  dsy=downsampley.get();
  dsz=downsamplez.get();
  if(dsx==1 && dsy == 1 && dsz == 1){
    outfield->send(ifield);
    return;
  }
  ofield=new ScalarFieldRG();
  ofield->resize((ifield->nx+dsx-1)/dsx, (ifield->ny+dsy-1)/dsy, (ifield->nz+dsz-1)/dsz);
  Point min, max;
  ifield->get_bounds(min, max);
  ofield->set_bounds(min, max);
  double mn, mx;
  ifield->get_minmax(mn, mx);
  ofield->set_minmax(mn, mx);
  ofield->xgrid=ifield->xgrid;
  ofield->ygrid=ifield->ygrid;
  ofield->zgrid=ifield->zgrid;
  np=Task::nprocessors();
  Task::multiprocess(np, do_parallel, this);
  outfield->send(ofield);
}
