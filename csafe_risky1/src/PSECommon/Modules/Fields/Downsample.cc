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

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Math/Expon.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/Thread.h>
#include <values.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Containers;
using namespace SCICore::Geometry;
using namespace SCICore::Math;
using namespace SCICore::Thread;

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
    virtual ~Downsample();
    virtual void execute();
};

extern "C" Module* make_Downsample(const clString& id) {
  return new Downsample(id);
}

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

Downsample::~Downsample()
{
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
  np=Thread::numProcessors();
  Thread::parallel(Parallel<Downsample>(this, &Downsample::parallel),
		   np, true);

  outfield->send(ofield);
}

} // End namespace Modules
} // End namespace PSECommon

