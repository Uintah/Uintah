/*
 *  Centroids.cc:
 *
 *  Written by:
 *   moulding
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/PointCloud.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <math.h>

#include <Core/share/share.h>

#include <vector>
#include <iostream>

namespace SCIRun {

using namespace std;

class PSECORESHARE Centroids : public Module {
public:
  Centroids(const string& id);
  virtual ~Centroids();
  virtual void execute();
};

extern "C" PSECORESHARE Module* make_Centroids(const string& id) {
  return scinew Centroids(id);
}

Centroids::Centroids(const string& id)
  : Module("Centroids", id, Source, "Fields", "SCIRun")
{
}

Centroids::~Centroids(){
}

void Centroids::execute()
{
  // must find ports and have valid data on inputs
  FieldIPort *ifieldPort = (FieldIPort*)get_iport("TetVol");

  if (!ifieldPort) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  FieldHandle ifieldH;
  if (!ifieldPort->get(ifieldH) || !ifieldH.get_rep()) return;
  MeshHandle ifieldMeshH = ifieldH->mesh();
  TetVolMesh *tvm = dynamic_cast<TetVolMesh*>(ifieldMeshH.get_rep());
  if (!tvm) {
    cerr << "Ceontroids error: input data wasn't a TetVol\n";
    return;
  }

  FieldOPort *ofieldPort = (FieldOPort*)get_oport("PointCloud");
  if (!ofieldPort) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  PointCloudMeshHandle pcm(scinew PointCloudMesh);
  TetVolMesh::Cell::iterator ci, ce;
  tvm->begin(ci); tvm->end(ce);
  Point p;
  while (ci != ce) {
    tvm->get_center(p, *ci);
    pcm->add_node(p);
    ++ci;
  }
  
  FieldHandle ofieldH(scinew PointCloud<double>(pcm, Field::NODE));
  ofieldPort->send(ofieldH);

}
} // End namespace SCIRun


