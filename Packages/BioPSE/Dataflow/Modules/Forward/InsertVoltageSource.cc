
/*
 *  InsertVoltageSource: Insert a voltage source
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

// check for same meshes as last time -- if so, reuse contrib array
// fix have_some -- just need to know which TVM indices we've visited

#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/TriSurf.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>

namespace BioPSE {

using namespace SCIRun;

class InsertVoltageSource : public Module {
  GuiInt outside_;
  GuiInt averaging_;
public:
  InsertVoltageSource(const string& id);
  virtual ~InsertVoltageSource();
  virtual void execute();
};

extern "C" Module* make_InsertVoltageSource(const string& id)
{
    return new InsertVoltageSource(id);
}

InsertVoltageSource::InsertVoltageSource(const string& id)
  : Module("InsertVoltageSource", id, Filter, "Forward", "BioPSE"),
    outside_("outside", id, this), averaging_("averaging", id, this)
{
}

InsertVoltageSource::~InsertVoltageSource()
{
}

void InsertVoltageSource::execute() {
  FieldIPort* imesh = (FieldIPort *) get_iport("TetMesh");
  FieldIPort* isource = (FieldIPort *) get_iport("TriSurfSource");
  FieldOPort* omesh = (FieldOPort *) get_oport("TetMesh");
  if (!imesh) {
    postMessage("Unable to initialize "+name+"'s imesh port\n");
    return;
  }
  if (!isource) {
    postMessage("Unable to initialize "+name+"'s isource port\n");
    return;
  }
  if (!omesh) {
    postMessage("Unable to initialize "+name+"'s omesh port\n");
    return;
  }
  
  FieldHandle imeshH;
  if (!imesh->get(imeshH))
    return;
  if (!imeshH.get_rep()) {
    cerr << "InsertVoltageSource: error - empty input mesh.\n";
    return;
  }

  MeshHandle tetVolH = imeshH->mesh();
  TetVolMesh *tvm = dynamic_cast<TetVolMesh *>(tetVolH.get_rep());
  if (!tvm) {
    cerr << "InsertVoltageSource: error - input FEM wasn't a TetVol\n";
    return;
  }

  FieldHandle isourceH;
  if (!isource->get(isourceH))
    return;
  if (!isourceH.get_rep()) {
    cerr << "InsertVoltageSource: error - empty input source.\n";
    return;
  }
  TriSurf<double> *triSurf =dynamic_cast<TriSurf<double>*>(isourceH.get_rep());
  if (!triSurf) {
    cerr << "InsertVoltageSource: error - input source wasn't a TtriSurf<double>\n";
    return;
  }
  TriSurfMesh *tsm = triSurf->get_typed_mesh().get_rep();

  vector<pair<int, double> > dirichlet;
  imeshH->get("dirichlet", dirichlet);
  vector<pair<int, double> > *new_dirichlet = 
    new vector<pair<int, double> >(dirichlet);

  vector<pair<string, Tensor> > conds;
  imeshH->get("conductivity_table", conds);
  vector<pair<string, Tensor> > *new_conds =
    new vector<pair<string, Tensor> >(conds);

  // get our own local copy of the Field and mesh
  imeshH.detach();  

  int averaging = averaging_.get();
  int outside = outside_.get();

  TetVolMesh::Node::size_type tvm_nnodes;
  TetVolMesh::Cell::size_type tvm_ncells;
  tvm->size(tvm_nnodes);
  tvm->size(tvm_ncells);

  TriSurfMesh::Node::iterator tsm_ni, tsm_ne;
  TriSurfMesh::Cell::iterator tsm_ci, tsm_ce;

  TetVolMesh::Node::array_type nbrs(4);

  Array1<Array1<pair<double, double> > > contrib;
  Array1<TetVolMesh::Node::index_type> have_some;
  tsm->begin(tsm_ni); tsm->end(tsm_ne);
  tsm->begin(tsm_ci); tsm->end(tsm_ce);

  // for each surface data_at position/value...
  while (tsm_ci != tsm_ce && tsm_ni != tsm_ne) {
    Point pt;
    double val;
    if (triSurf->data_at() == Field::CELL) {
      tsm->get_center(pt, *tsm_ci);
      val = triSurf->fdata()[*tsm_ci];
      ++tsm_ci;
    } else {
      tsm->get_center(pt, *tsm_ni);
      val = triSurf->fdata()[*tsm_ni];
      ++tsm_ni;
    }

    // find the tet nodes (nbrs) that it's closest to
    TetVolMesh::Cell::index_type tvm_cidx;
    if (tvm->locate(tvm_cidx, pt)) {
      tvm->get_nodes(nbrs, tvm_cidx);
    } else if (outside) {
      tvm->locate(nbrs[0], pt);
      nbrs.resize(1);
    } else continue;
  
    // for each nbr, store the value/distance in contrib[nbr]
    unsigned int i;
    for (i=0; i<nbrs.size(); i++) {
      Point nbr_pt;
      tvm->get_center(nbr_pt, nbrs[i]);
      double d = (pt - nbr_pt).length();
      if (d < 1.e-6) d=1.e-6;
      d = 1/d;
      pair<double, double> p(d, val);
      have_some.add(nbrs[i]);
      if (contrib[nbrs[i]].size() == 0) {
	if (averaging) {
	  contrib[nbrs[i]].add(p);
	} else {
	  if (contrib[nbrs[i]].size() == 0) {
	    contrib[nbrs[i]].add(p);
	  } else if (d > contrib[nbrs[i]][0].first) {
	    contrib[nbrs[i]][0] = p;
	  }
	}
      }
    }
  }

  // for each tet node that was touched, compute 1/(sum of all the distances)
  // multiply each distance by that to get weightings
  int i,j;
  for (i=0; i<have_some.size(); i++) {
    double sum_dist = 0;
    TetVolMesh::Node::index_type node = have_some[i];
    for (j=0; j<contrib[node].size(); j++) {
      sum_dist += contrib[node][j].first;
    }
    sum_dist = 1./sum_dist;
    for (j=0; j<contrib[node].size(); j++) {
      contrib[node][j].first *= sum_dist;
    }
  }

  // if no meshes changed, we'll use the same weightings -- just new values
  for (i=0; i<have_some.size(); i++) {
    TetVolMesh::Node::index_type node = have_some[i];
    double avg_val = 0;
    for (j=0; j<contrib[node].size(); j++) {
      avg_val += contrib[node][j].second * contrib[node][j].first;
    }
    new_dirichlet->push_back(pair<int, double>(node, avg_val));
  }
  imeshH->store("dirichlet", *new_dirichlet, false);
  imeshH->store("conductivity_table", *new_conds, false);
  omesh->send(imeshH);
}
} // End namespace BioPSE
