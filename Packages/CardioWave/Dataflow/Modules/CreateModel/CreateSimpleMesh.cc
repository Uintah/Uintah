/*
 *  CreateSimpleMesh.cc:
 *
 *   Written by:
 *   Joe Tranquillo
 *   Duke University 
 *   Biomedical Engineering Department
 *   August 2001
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Dataflow/Ports/FieldPort.h>

#include <Packages/CardioWave/share/share.h>

namespace CardioWave {

using namespace SCIRun;

class CardioWaveSHARE CreateSimpleMesh : public Module {
	GuiInt	xdim_;
	GuiInt	ydim_;
	GuiInt	zdim_;
	GuiDouble	dx_;
	GuiDouble	dy_;
	GuiDouble	dz_;
	GuiDouble	fib1x_;
	GuiDouble	fib1y_;
	GuiDouble	fib1z_;
	GuiDouble	fib2x_;
	GuiDouble	fib2y_;
	GuiDouble	fib2z_;

public:
  CreateSimpleMesh(const string& id);
  virtual ~CreateSimpleMesh();
  virtual void execute();
};

extern "C" CardioWaveSHARE Module* make_CreateSimpleMesh(const string& id) {
  return scinew CreateSimpleMesh(id);
}

CreateSimpleMesh::CreateSimpleMesh(const string& id)
  : Module("CreateSimpleMesh", id, Source, "CreateModel", "CardioWave"),
    xdim_("xdim", id, this),
    ydim_("ydim", id, this),
    zdim_("zdim", id, this),
    dx_("dx", id, this),
    dy_("dy", id, this),
    dz_("dz", id, this),
    fib1x_("fib1x", id, this),
    fib1y_("fib1y", id, this),
    fib1z_("fib1z", id, this),
    fib2x_("fib2x", id, this),
    fib2y_("fib2y", id, this),
    fib2z_("fib2z", id, this)   
{
}

CreateSimpleMesh::~CreateSimpleMesh(){
}


void CreateSimpleMesh::execute(){



  int xdim = xdim_.get();
  int ydim = xdim_.get();
  int zdim = xdim_.get();
  double dx = dx_.get();
  double dy = dy_.get();
  double dz = dz_.get();
  double fib1x = fib1x_.get();
  double fib1y = fib1y_.get();
  double fib1z = fib1z_.get();
  double fib2x = fib2x_.get();
  double fib2y = fib2y_.get();
  double fib2z = fib2z_.get();

  LatVolMesh *mesh = scinew LatVolMesh(xdim, ydim, zdim, Point(0,0,0),
				       Point((xdim-1)*dx, (ydim-1)*dy,
					     (ydim-1)*dy));
  LatticeVol<int> *fld = scinew LatticeVol<int>(mesh, Field::NODE);
  Vector v1(fib1x, fib1y, fib1z);
  if (!v1.length()) {
    cerr << "Error -- fib1 was zero length\n";
    return;
  }
  v1.normalize();
  Vector v2(fib2x, fib2y, fib2z);
  if (!v2.length()) {
    cerr << "Error -- fib2 was zero length\n";
    return;
  }
  v2.normalize();
  Vector v3 = Cross(v1,v2);
  if (!v3.length()) {
    cerr << "Error -- fib1 and fib2 need to be in different directions!\n";
    return;
  }
  if (v3.length() < .99 || v3.length() > 1.01) {
    cerr << "Corrected fib2 to make it orthogonal to fib1.\n";
  }

  v3.normalize();
  v2 = Cross(v3, v1);

  int i, j, k;
  for (i=0; i<xdim; i++)
    for (j=0; j<ydim; j++)
      for (k=0; k<zdim; k++) 
	fld->fdata()(i,j,k)=0;

  Array1<Array1<Vector> > fibers;
  fibers.resize(1);
  fibers[0].add(v1); fibers[0].add(v2); fibers[0].add(v3);
  fld->store("eigenvectors", fibers);
  FieldOPort *ofield_port = (FieldOPort *)get_oport("Mesh");
  if (!ofield_port) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  ofield_port->send(FieldHandle(fld));
}


} // End namespace CardioWave


