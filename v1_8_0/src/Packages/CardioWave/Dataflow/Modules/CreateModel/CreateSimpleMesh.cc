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
#include <Core/Datatypes/LatVolField.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/Array1.h>

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
  CreateSimpleMesh(GuiContext *context);
  virtual ~CreateSimpleMesh();
  virtual void execute();
};


DECLARE_MAKER(CreateSimpleMesh)


CreateSimpleMesh::CreateSimpleMesh(GuiContext *context)
  : Module("CreateSimpleMesh", context, Source, "CreateModel", "CardioWave"),
    xdim_(context->subVar("xdim")),
    ydim_(context->subVar("ydim")),
    zdim_(context->subVar("zdim")),
    dx_(context->subVar("dx")),
    dy_(context->subVar("dy")),
    dz_(context->subVar("dz")),
    fib1x_(context->subVar("fib1x")),
    fib1y_(context->subVar("fib1y")),
    fib1z_(context->subVar("fib1z")),
    fib2x_(context->subVar("fib2x")),
    fib2y_(context->subVar("fib2y")),
    fib2z_(context->subVar("fib2z"))
{
}

CreateSimpleMesh::~CreateSimpleMesh(){
}


void
CreateSimpleMesh::execute()
{
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
					     (zdim-1)*dz));
  LatVolField<int> *fld = scinew LatVolField<int>(mesh, Field::NODE);
  Vector v1(fib1x, fib1y, fib1z);
  if (!v1.length()) {
    error("fib1 was zero length.");
    return;
  }
  v1.normalize();
  Vector v2(fib2x, fib2y, fib2z);
  if (!v2.length()) {
    error("fib2 was zero length.");
    return;
  }
  v2.normalize();
  Vector v3 = Cross(v1,v2);
  if (!v3.length()) {
    error("fib1 and fib2 need to be in different directions!");
    return;
  }
  if (v3.length() < .99 || v3.length() > 1.01) {
    remark("Corrected fib2 to make it orthogonal to fib1.");
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
  fld->set_property("eigenvectors", fibers, false);
  FieldOPort *ofield_port = (FieldOPort *)get_oport("Mesh");
  if (!ofield_port) {
    error("Unable to initialize oport 'Mesh'.");
    return;
  }
  ofield_port->send(FieldHandle(fld));
}


} // End namespace CardioWave


