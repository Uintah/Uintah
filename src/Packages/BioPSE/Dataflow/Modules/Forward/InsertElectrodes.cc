
/*
 *  InsertElectrodes: Insert an electrode into a finite element mesh
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/Datatypes/ContourField.h>
#include <Core/Datatypes/TetVol.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>

namespace BioPSE {

using namespace SCIRun;

class InsertElectrodes : public Module {
  void insertContourIntoTetMesh(vector<pair<int, double> > &dirichlet,
				TetVolMeshHandle mesh,
				Array1<Point> &inner,
				Array1<Point> &outer,
				double voltage);
  bool point_in_loop(const Point &pt, 
		     const Array1<Point> &contour, 
		     const Plane &pl);

public:
  InsertElectrodes(const string& id);
  virtual ~InsertElectrodes();
  virtual void execute();
};

extern "C" Module* make_InsertElectrodes(const string& id)
{
    return new InsertElectrodes(id);
}

InsertElectrodes::InsertElectrodes(const string& id)
: Module("InsertElectrodes", id, Filter, "Forward", "BioPSE")
{
}

InsertElectrodes::~InsertElectrodes()
{
}

bool InsertElectrodes::point_in_loop(const Point &pt,
				    const Array1<Point> &contour,
				    const Plane &pl) {
  int sign=1;
  if (pl.eval_point(pt+(Cross(pt-contour[0], pt-contour[1]))) < 0) sign=-1;
  int i;
  for (i=1; i<contour.size()-1; i++)
    if (pl.eval_point(pt+(Cross(pt-contour[i], pt-contour[i+1])))*sign < 0)
      return false;
  if (pl.eval_point(pt+(Cross(pt-contour[i], pt-contour[0])))*sign < 0)
    return false;
  return true;
}

void InsertElectrodes::insertContourIntoTetMesh(
			        vector<pair<int, double> > &dirichlet,
				TetVolMeshHandle tet_mesh,
				Array1<Point> &inner,
				Array1<Point> &outer,
				double voltage) {
//  cerr << "entering insertCountourIntoTetMesh\n";
  Plane electrode_plane;
  if (Cross(inner[1]-inner[0], inner[1]-outer[0]).length2()>1.e-5) {
    electrode_plane=Plane(inner[1], inner[0], outer[0]);
  } else {
    electrode_plane=Plane(inner[2], inner[0], outer[0]);
  }


  TetVolMesh::Cell::size_type ncells;
  tet_mesh->size(ncells);
  TetVolMesh::Node::size_type nnodes;
  tet_mesh->size(nnodes);

  TetVolMesh::Node::array_type tet_nodes;
  Array1<int> is_electrode_node(nnodes); // -1:unknow, 0:no, 1:outer, 2:inner
  is_electrode_node.initialize(-1);
  Array1<TetVolMesh::Node::index_type> electrode_nodes;
  Array1<TetVolMesh::Node::index_type> electrode_node_split_idx(nnodes);

  TetVolMesh::Cell::iterator tet_cell_idx, tet_cell_idx_end;
  tet_mesh->begin(tet_cell_idx); tet_mesh->end(tet_cell_idx_end);
  while (tet_cell_idx != tet_cell_idx_end) {
    tet_mesh->get_nodes(tet_nodes, *tet_cell_idx);
    int all_nodes_above_plane=1, all_nodes_below_plane=1;
    Array1<Point> tet_node_pts(4);
    double signed_distance[4];
    Array1<TetVolMesh::Node::index_type> projections;
    for (int ii=0; ii<4; ii++) {
      tet_mesh->get_center(tet_node_pts[ii], tet_nodes[ii]);
      signed_distance[ii]=electrode_plane.eval_point(tet_node_pts[ii]);
      if (signed_distance[ii] > 0) {
	all_nodes_above_plane = 0;
      } else {
	all_nodes_below_plane = 0;
	projections.add(tet_nodes[ii]);
      }
    }
    if (!all_nodes_below_plane && 
	!all_nodes_above_plane) { // cell was intersected by plane...
      
      // check to see if projected nodes are inside the electrode
      int inside_inner=0, inside_outer=0, idx;
      for (idx=0; idx<projections.size(); idx++) {
	Point p;
	TetVolMesh::Node::index_type nidx(projections[idx]);
	tet_mesh->get_center(p, nidx);
	if (point_in_loop(p, inner, electrode_plane)) {
	  inside_inner = 1;
	  if (is_electrode_node[nidx] == -1) {
	    is_electrode_node[nidx] = 2;
	    electrode_nodes.add(nidx);
	  }
	}
      }
      for (idx=0; idx<projections.size(); idx++) {
	Point p;
	TetVolMesh::Node::index_type nidx(projections[idx]);
	if (point_in_loop(p, outer, electrode_plane)) {
	  inside_outer = 1;
	  if (is_electrode_node[nidx] == -1) {
	    is_electrode_node[nidx] = 1;
	    electrode_nodes.add(nidx);
	  }
	}
      }
    }
    ++tet_cell_idx;
  }

  // project electrode nodes to plane, split them, and set Dirichlet
  int i;
  cerr << "Electrode nodes: \n";
  for (i=0; i<electrode_nodes.size(); i++) {
    Point pt;
    TetVolMesh::Node::index_type ni = electrode_nodes[i];
    tet_mesh->get_center(pt, ni);
    Point proj_pt = electrode_plane.project(pt);
    tet_mesh->set_point(proj_pt, ni);
    electrode_node_split_idx[ni] = tet_mesh->add_point(proj_pt);
    cerr << "  "<<i<<" index="<<(int)(ni) <<" (copy index="<<(int)(electrode_node_split_idx[ni])<<"), is_elec_nodes="<<is_electrode_node[electrode_nodes[i]]<<"\n";
    if (is_electrode_node[electrode_nodes[i]] == 2) {
//      dirichlet.push_back(pair<int,double>(electrode_node_split_idx[ni], 
//					   voltage));
      dirichlet.push_back(pair<int,double>(ni, voltage));
    }
  }

  // set elements above the plane to point to new split node
  TetVolMesh::Cell::iterator ci, ce;
  tet_mesh->begin(ci); tet_mesh->end(ce);
  while (ci != ce) {
    tet_mesh->get_nodes(tet_nodes, *ci);
    int have_any =0 ;
    for (i=0; i<4; i++) {
      if (is_electrode_node[tet_nodes[i]] > 0) have_any = 1;
    }
    if (have_any) {
//      cerr << "Touching element ("<<(int)(*ci)<<") ";
      Point centroid;
      tet_mesh->get_center(centroid, *ci);
      double centroid_dist = electrode_plane.eval_point(centroid);
      if (centroid_dist>0) {
//	cerr << "above plane";
	int remap=0;
	for (i=0; i<4; i++) {
	  if (is_electrode_node[tet_nodes[i]] > 0) {
	    remap++;
	    tet_nodes[i] = electrode_node_split_idx[tet_nodes[i]];
	  }
	}
//	cerr << " ("<<remap<<")";
	tet_mesh->set_nodes(tet_nodes, *ci);
      }
//      cerr << "\n";
    }
    ++ci;
  }
  tet_mesh->recompute_connectivity();
}

void InsertElectrodes::execute() {
  FieldIPort* imesh = (FieldIPort *) get_iport("TetMesh");
  FieldOPort* omesh = (FieldOPort *) get_oport("TetMesh");
  if (!imesh) {
    postMessage("Unable to initialize "+name+"'s imesh port\n");
    return;
  }
  if (!omesh) {
    postMessage("Unable to initialize "+name+"'s omesh port\n");
    return;
  }
  
  FieldHandle imeshH, ielecH;

  if (!imesh->get(imeshH))
    return;
  if (!imeshH.get_rep()) {
    cerr << "InsertElectrodes: error - empty input mesh.\n";
    return;
  }
  if (!(dynamic_cast<TetVol<int>*>(imeshH.get_rep()))) {
    cerr << "InsertElectrodes: error - input FEM wasn't a TetVol<int>\n";
    return;
  }

  vector<pair<int, double> > dirichlet_nodes;
  imeshH->get("dirichlet", dirichlet_nodes);
  vector<pair<string, Tensor> > conds;
  imeshH->get("conductivity_table", conds);
  vector<pair<string, Tensor> > *newConds =
    new vector<pair<string, Tensor> >(conds);
  for (int ii=0; ii<newConds->size(); ii++) {
    cerr << "New conds ["<<ii<<"] = "<<(*newConds)[ii].first<<" , "<<(*newConds)[ii].second<<"\n";
  }

  vector<pair<int, double> > *mergedDirichletNodes = 
    new vector<pair<int, double> >(dirichlet_nodes);

  port_range_type range = get_iports("Electrodes");
  if (range.first != range.second) {
    cerr << "Cloning data...\n";
    // get our own local copy of the Field and mesh
    imeshH.detach();
    cerr << "Cloning mesh...\n";
    imeshH->mesh_detach();
    cerr << "Done cloning!\n";

    TetVol<int> *field = dynamic_cast<TetVol<int>*>(imeshH.get_rep());
    TetVolMeshHandle mesh = field->get_typed_mesh();

    port_map_type::iterator pi = range.first;
    while (pi != range.second) {
      FieldIPort *ielec = (FieldIPort *)get_iport(pi->second);
      if (!ielec->get(ielecH) || !ielecH.get_rep()) {
	++pi;
	continue;
      }

      ContourField<double> *elecFld = 
	dynamic_cast<ContourField<double>*>(ielecH.get_rep());
      if (!elecFld) {
	cerr << "InsertElectrodes: error - input electrode wasn't a ContourField<double>\n";
	++pi;
	continue;
      }
      double voltage = elecFld->fdata()[0];
      ContourMeshHandle elecMesh(elecFld->get_typed_mesh());
      Array1<Point> outer;
      Array1<Point> inner;
      ContourMesh::Node::iterator ni, ne;
      elecMesh->begin(ni);
      elecMesh->end(ne);
      for (int i=0; i<6; i++) ++ni; // the first six nodes are the handle
      while(ni != ne) {
	Point p;
	elecMesh->get_center(p, *ni);
	if (elecFld->fdata()[*ni] == 0) inner.add(p);
	else outer.add(p);
	++ni;
      }
      
      // modify mergedDirichletNodes, and mergedMesh to have new nodes
      insertContourIntoTetMesh(*mergedDirichletNodes, mesh,
			       inner, outer, voltage);
      
      ++pi;
    }
    imeshH->store("dirichlet", *mergedDirichletNodes, false);
    imeshH->store("conductivity_table", *newConds, false);
    omesh->send(imeshH);
  } else {
    cerr << "Error - unable to get electrode.\n";
    omesh->send(imeshH);
  }
}
} // End namespace BioPSE
