/*
 *  SetupFVM2.cc:
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
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/HexVolField.h>

extern "C" {
#include <Packages/CardioWave/Core/Algorithms/Vulcan.h>
}

#include <Packages/CardioWave/share/share.h>

namespace CardioWave {

using namespace SCIRun;

class CardioWaveSHARE SetupFVM2 : public Module {
  GuiDouble	bathsig_;
  GuiDouble	fibersig1_;
  GuiDouble	fibersig2_;
  GuiString	sprfile_;
  GuiString	volumefile_;
  GuiString	visfile_;
  GuiString	idfile_;
  
public:
  SetupFVM2(GuiContext *context);
  virtual ~SetupFVM2();
  virtual void execute();
};


DECLARE_MAKER(SetupFVM2)


SetupFVM2::SetupFVM2(GuiContext *context)
  : Module("SetupFVM2", context, Source, "CreateModel", "CardioWave"),
    bathsig_(context->subVar("bathsig")),
    fibersig1_(context->subVar("fibersig1")),
    fibersig2_(context->subVar("fibersig2")),
    sprfile_(context->subVar("sprfile")),
    volumefile_(context->subVar("volumefile")),
    visfile_(context->subVar("visfile")),
    idfile_(context->subVar("idfile"))
{
}

SetupFVM2::~SetupFVM2(){
}

void SetupFVM2::execute(){
  double bathsig = bathsig_.get();
  double fibersig1 = fibersig1_.get();
  double fibersig2 = fibersig2_.get();
  string sprfile = sprfile_.get();
  string volumefile = volumefile_.get();
  string visfile = visfile_.get();
  string idfile = idfile_.get();
  
  // must find ports and have valid data on inputs
  FieldIPort *ifld = (FieldIPort*)get_iport("HexVolTensor");
  if (!ifld) {
    error("Unable to initialize iport 'HexVolTensor'.");
    return;
  }

  FieldHandle fldH;
  if (!ifld->get(fldH) || 
      !fldH.get_rep())
  {
    error("Empty input field.");
    return;
  }

  HexVolField<int> *fld = dynamic_cast<HexVolField<int> *>(fldH.get_rep());
  if (!fld) {
    error("Input field wasn't a HexVolField<int>.");
    return;
  }
  
  vector<pair<string, Tensor> > tens;
  if (!fld->get_property("conductivity_table", tens)) {
    error("No tensor vector associated with the input field.");
    return;
  }
  
  HexVolMeshHandle m = fld->get_typed_mesh();
  HexVolMesh::Node::size_type nnodes;
  HexVolMesh::Cell::size_type ncells;
  m->size(nnodes);
  m->size(ncells);

  msgStream_ << "nnodes="<<nnodes<<" ncells="<<ncells<<"\n";

  MESH *mesh = new MESH;
  mesh->vtx = new VERTEX[nnodes];
  mesh->numvtx = nnodes;
  mesh->elements = new VOLUME_ELEMENT[ncells];
  mesh->numelement = ncells;
  
  // fill in mesh from SCIRun field

  HexVolMesh::Node::iterator nb, ne; m->begin(nb); m->end(ne);
  int cnt=0;
  Point p;
  FILE *IDFILE=0;
  if (idfile != "") {
    IDFILE = fopen(idfile.c_str(), "w");
    fprintf(IDFILE, "%d\n", (int)nnodes);
  }

  while(nb != ne) {
    int tidx;
    fld->value(tidx, *nb);
    m->get_center(p, *nb);
    mesh->vtx[cnt].x = p.x();
    mesh->vtx[cnt].y = p.y();
    mesh->vtx[cnt].z = p.z();

    if (tidx == 0 || tidx == 1) {
      if (IDFILE) fprintf(IDFILE, "0\n");
      mesh->vtx[cnt].sxx = bathsig;
      mesh->vtx[cnt].sxy = 0;
      mesh->vtx[cnt].sxz = 0;
      mesh->vtx[cnt].syy = bathsig;
      mesh->vtx[cnt].syz = 0;
      mesh->vtx[cnt].szz = bathsig;
      mesh->vtx[cnt].volume=0;
    } else { // assuming type = 1
      if (IDFILE) fprintf(IDFILE, "1\n");
      Tensor t = tens[tidx].second;
      Vector f1, f2, f3;
      t.get_eigenvectors(f1, f2, f3);
      mesh->vtx[cnt].sxx = f1.x()*fibersig1*f1.x() + 
                              f1.y()*fibersig2*f1.y() + 
                              f1.z()*fibersig2*f1.z();
      mesh->vtx[cnt].sxy = f1.x()*fibersig1*f2.x() + 
                              f1.y()*fibersig2*f2.y() + 
                              f1.z()*fibersig2*f2.z();
      mesh->vtx[cnt].sxz = f1.x()*fibersig1*f3.x() + 
                              f1.y()*fibersig2*f3.y() + 
                              f1.z()*fibersig2*f3.z();
      mesh->vtx[cnt].syy = f2.x()*fibersig1*f2.x() + 
                              f2.y()*fibersig2*f2.y() + 
                              f2.z()*fibersig2*f2.z();
      mesh->vtx[cnt].syz = f2.x()*fibersig1*f3.x() + 
                              f2.y()*fibersig2*f3.y() + 
                              f2.z()*fibersig2*f3.z();
      mesh->vtx[cnt].szz = f3.x()*fibersig1*f3.x() + 
                              f3.y()*fibersig2*f3.y() + 
                              f3.z()*fibersig2*f3.z();
      mesh->vtx[cnt].volume=0;
    }
    cnt++;
    ++nb;
  }
  if (IDFILE) fclose(IDFILE);

  HexVolMesh::Cell::iterator cb, ce; m->begin(cb); m->end(ce);
  HexVolMesh::Node::array_type nodes;
  cnt=0;

  while(cb != ce) {
    m->get_nodes(nodes, *cb);
    mesh->elements[cnt].vtx[0] = nodes[0];
    mesh->elements[cnt].vtx[1] = nodes[1];
    mesh->elements[cnt].vtx[2] = nodes[2];
    mesh->elements[cnt].vtx[3] = nodes[3];
    mesh->elements[cnt].vtx[4] = nodes[4];
    mesh->elements[cnt].vtx[5] = nodes[5];
    mesh->elements[cnt].vtx[6] = nodes[6];
    mesh->elements[cnt].vtx[7] = nodes[7];
    cnt++;
    ++cb;
  }
    
  compute_volumes(mesh, volumefile.c_str());
  compute_matrix(mesh, sprfile.c_str());
  dump_vis(mesh, visfile.c_str());
}

} // End namespace CardioWave
