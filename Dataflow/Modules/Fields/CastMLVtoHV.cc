/*
 *  CastMLVtoHM.cc:
 *
 *  Written by:
 *   moulding
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/HexVol.h>
#include <Core/Datatypes/MaskedLatticeVol.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <math.h>

#include <Core/share/share.h>

#include <vector>
#include <iostream>

namespace SCIRun {

using namespace std;

class PSECORESHARE CastMLVtoHM : public Module {
private:
  int            last_gen_;
  FieldHandle    ofieldH_;
public:
  CastMLVtoHM(const string& id);
  virtual ~CastMLVtoHM();
  virtual void execute();
};

extern "C" PSECORESHARE Module* make_CastMLVtoHM(const string& id) {
  return scinew CastMLVtoHM(id);
}

CastMLVtoHM::CastMLVtoHM(const string& id)
  : Module("CastMLVtoHM", id, Source, "Fields", "SCIRun"), last_gen_(-1)
{
}

CastMLVtoHM::~CastMLVtoHM(){
}

void CastMLVtoHM::execute()
{
  // must find ports and have valid data on inputs
  FieldIPort *iport_ = (FieldIPort*)get_iport("MaskedLatticeVol");

  if (!iport_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  
  FieldOPort *oport_ = (FieldOPort*)get_oport("HexVol");
  if (!oport_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  FieldHandle ifieldH;
  if (!iport_->get(ifieldH) || 
      !ifieldH.get_rep())
    return;
  
  if (ifieldH->generation == last_gen_) {
    oport_->send(ofieldH_);
    return;
  }
  last_gen_ = ifieldH->generation;

  // we expect that the input field is a TetVol<Vector>
  if (ifieldH->get_type_description()->get_name() !=
      get_type_description((MaskedLatticeVol<Vector> *)0)->get_name())
  {
    postMessage("CastMLVtoHM: ERROR: input volume is not a MaskedLatticeVol<Vector>.  Exiting.");
    return;
  }                     

  if (ifieldH->data_at() != Field::NODE) {
    postMessage("CastMLVtoHM: ERROR: input volume data isn't node-centered.  Existing.");
    return;
  }                         

  MaskedLatticeVol<Vector> *lv = dynamic_cast<MaskedLatticeVol<Vector> *>(ifieldH.get_rep());
  LatVolMesh *lvm = lv->get_typed_mesh().get_rep();
  HexVolMesh *hvm = scinew HexVolMesh;

  // fill in the nodes and connectivities
  Point min = lvm->get_min();
  Vector diag = lvm->diagonal();
  int nx = lvm->get_nx();
  int ny = lvm->get_ny();
  int nz = lvm->get_nz();
  double dx = diag.x()/(nx-1);
  double dy = diag.y()/(ny-1);
  double dz = diag.z()/(nz-1);

  int i, j, k;
  int ii, jj, kk;
  Array3<int> connectedNodes(nz, ny, nx);
  connectedNodes.initialize(0);
  for (k=0; k<nz-1; k++)
    for (j=0; j<ny-1; j++)
      for (i=0; i<nx-1; i++) {
	int valid=1;
	for (ii=0; ii<2; ii++)
	  for (jj=0; jj<2; jj++)
	    for (kk=0; kk<2; kk++)
	      if (!lv->mask()(k+kk,j+jj,i+ii)) valid=0;
	if (valid)
	  for (ii=0; ii<2; ii++)
	    for (jj=0; jj<2; jj++)
	      for (kk=0; kk<2; kk++)
		connectedNodes(k+kk,j+jj,i+ii)=1;
      }

  Array3<int> nodeMap(nz, ny, nx);
  nodeMap.initialize(-1);
  Vector dummy;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++)
	if (connectedNodes(k,j,i))
	  nodeMap(k,j,i) = hvm->add_point(min + Vector(dx*i, dy*j, dz*k));

  for (k=0; k<nz-1; k++)
    for (j=0; j<ny-1; j++)
      for (i=0; i<nx-1; i++) {
	  HexVolMesh::Node::index_type n000, n001, n010, n011, n100, n101, n110, n111;
	  if ((n000 = nodeMap(k  , j  , i  )) == -1) continue;
	  if ((n001 = nodeMap(k  , j  , i+1)) == -1) continue;
	  if ((n010 = nodeMap(k  , j+1, i  )) == -1) continue;
	  if ((n011 = nodeMap(k  , j+1, i+1)) == -1) continue;
	  if ((n100 = nodeMap(k+1, j  , i  )) == -1) continue;
          if ((n101 = nodeMap(k+1, j  , i+1)) == -1) continue;
	  if ((n110 = nodeMap(k+1, j+1, i  )) == -1) continue;
	  if ((n111 = nodeMap(k+1, j+1, i+1)) == -1) continue;
	  hvm->add_hex(n000, n001, n011, n010, n100, n101, n111, n110);
      }
      
  HexVol<Vector> *hv = scinew HexVol<Vector>(hvm, Field::NODE);
  int count=0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++)
	if (lv->mask()(k,j,i))
	  hv->fdata()[count++] = lv->fdata()(k,j,i);

  ofieldH_ = hv;
  oport_->send(ofieldH_);
}

} // End namespace SCIRun


