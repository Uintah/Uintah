/*
 *  MeshGen.cc:
 *
 *  Written by:
 *   wangxl
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/Wangxl/share/share.h>

#include <Dataflow/Ports/FieldPort.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/Delaunay.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/FObject.h>

#include <fstream>

namespace Wangxl {

using namespace SCIRun;

class WangxlSHARE MeshGen : public Module {
  TriSurfMesh *d_sMesh;
  Delaunay d_vMesh;
  vector<DVertex*> d_vertices;
  hash_map<DVertex*, int, MHash, MEqual> d_vmap;
  double d_size;
  int d_index; // pointer to the current vertex

  FieldIPort* d_iport;
  FieldOPort* d_oport;
public:
  MeshGen(GuiContext*);

  virtual ~MeshGen();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
private:
  void get_data();
};


DECLARE_MAKER(MeshGen)
MeshGen::MeshGen(GuiContext* ctx)
  : Module("MeshGen", ctx, Source, "Volume", "Wangxl")
{
}

MeshGen::~MeshGen(){
}

void
 MeshGen::execute(){
  d_iport = (FieldIPort *)get_iport("surface_mesh");
  d_oport = (FieldOPort *)get_oport("volume_mesh");

  if (!d_iport) {
    error("Unable to initialize iport 'point_cloud'.");
    return;
  }
  if (!d_oport) {
    error("Unable to initialize iport 'surface_mesh'.");
    return;
  }
  get_data();

}

void MeshGen::get_data() {
  DVertex* v;
  Field *iField;
  FieldHandle iHandle;
  TriSurfField<double> *sField;
  node_iterator ni, ni_end;
  Point point;

  d_iport = (FieldIPort*)get_iport("TriSurfField");
  if (!d_iport->get(iHandle) || !(iField = iHandle.get_rep())) return;
  sField = (TriSurfField<double>*)iField;
  d_sMesh =  sField->get_typed_mesh().get_rep();
  
  d_index = 0;
  d_sMesh->begin(ni);
  d_sMesh->end(ni_end);
  d_vertices.clear();
  while ( ni != ni_end ) {
    d_sMesh->get_point(point,*ni);
    v = d_vMesh.insert(point);
    v->set_input(true); // assign a flag which means this vertex is the original input
    d_vertices.push_back(v);
    d_vmap[v] = d_index++;
    ++ni;
  }
  /*  ifstream ptsFile("data/tank0.pts");
  
  double x, y, z;
  while ( ptsFile >> x >> y >> z ) {
    v = d_vMesh.insert(Point(x,y,z));
    v->set_input(true); // all points are original
    d_vertices.push_back(v);
    d_vmap[v] = d_index++;
  }
  d_vMesh.is_valid(true,0);*/
}


void
 MeshGen::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Wangxl


