
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

#include <Core/Datatypes/CurveField.h>
#include <Core/Datatypes/TetVolField.h>
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
				double voltage,
				TetVolMesh* electrodeElements);
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
  Vector v1, v2, v3;
  int i;
  double angle, sum=0;
  for (i=0; i<contour.size()-1; i++) {
    v1=pt-contour[i];
    v2=pt-contour[i+1];
    v1.normalize();
    v2.normalize();
    angle=acos(Dot(v1,v2));
    v3=Cross(v1, v2);
    if (pl.eval_point(pt+v3) < 0) angle=-angle;
    sum += angle;
  }
  sum/=(2.*M_PI);
  sum=fabs(sum);
  sum -= (((int)sum)/2)*2;
  // sum is now between 0 and 1.99999
  if (sum > 0.8 && sum < 1.2) return true;
  else if (sum < 0.2 || sum > 1.8) return false;
  else {
    cerr << "Error -- point in loop was: "<<sum<<"\n";
    return false;
  }
}

void InsertElectrodes::insertContourIntoTetMesh(
			        vector<pair<int, double> > &dirichlet,
				TetVolMeshHandle tet_mesh,
				Array1<Point> &inner,
				Array1<Point> &outer,
				double voltage,
				TetVolMesh* electrodeElements) {
//  cerr << "entering insertCountourIntoTetMesh\n";
  Plane electrode_plane;
  if (Cross(inner[1]-inner[0], inner[1]-inner[2]).length2()>1.e-5) {
    electrode_plane=Plane(inner[1], inner[0], inner[2]);
  } else if (Cross(inner[1]-inner[0], inner[1]-inner[3]).length2()>1.e-5) {
    electrode_plane=Plane(inner[1], inner[0], inner[3]);
  } else {
    cerr << "InsertElectrode ERROR - FIRST FOUR INNER NODES OF THE ELEC ARE COLINEAR!\n";
    return;
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
      int idx;
      for (idx=0; idx<projections.size(); idx++) {
	Point p;
	TetVolMesh::Node::index_type nidx(projections[idx]);
	tet_mesh->get_center(p, nidx);
	p = electrode_plane.project(p);
	if (point_in_loop(p, inner, electrode_plane)) {
	  if (is_electrode_node[nidx] == -1) {
	    is_electrode_node[nidx] = 2;
	    electrode_nodes.add(nidx);
	  }
	}
      }
      for (idx=0; idx<projections.size(); idx++) {
	Point p;
	TetVolMesh::Node::index_type nidx(projections[idx]);
	tet_mesh->get_center(p, nidx);
	p = electrode_plane.project(p);
	if (point_in_loop(p, outer, electrode_plane)) {
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
//  cerr << "Electrode nodes: \n";
  for (i=0; i<electrode_nodes.size(); i++) {
    Point pt;
    TetVolMesh::Node::index_type ni = electrode_nodes[i];
    tet_mesh->get_center(pt, ni);
    Point proj_pt = electrode_plane.project(pt);
    tet_mesh->set_point(proj_pt, ni);
    electrode_node_split_idx[ni] = tet_mesh->add_point(proj_pt);
//    cerr << "  "<<i<<" index="<<(int)(ni) <<" (copy index="<<(int)(electrode_node_split_idx[ni])<<"), is_elec_nodes="<<is_electrode_node[electrode_nodes[i]]<<"\n";
    if (is_electrode_node[electrode_nodes[i]] == 2) {
      dirichlet.push_back(pair<int,double>(electrode_node_split_idx[ni], 
					   voltage));
//      dirichlet.push_back(pair<int,double>(ni, voltage));
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
	if (electrodeElements) {
	  Point pts[4];
	  for (i=0; i<4; i++) tet_mesh->get_point(pts[i], tet_nodes[i]);
	  electrodeElements->add_tet_unconnected(pts[0], pts[1], pts[2], pts[3]);
	}
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
  FieldOPort* oelec = (FieldOPort *) get_oport("ElectrodeElements");
  if (!oelec) {
    postMessage("Unable to initialize "+name+"'s oelec port\n");
    return;
  }
  
  FieldHandle imeshH, ielecH;

  if (!imesh->get(imeshH))
    return;
  if (!imeshH.get_rep()) {
    cerr << "InsertElectrodes: error - empty input mesh.\n";
    return;
  }
  if (!(dynamic_cast<TetVolField<int>*>(imeshH.get_rep()))) {
    cerr << "InsertElectrodes: error - input FEM wasn't a TetVolField<int>\n";
    return;
  }

  vector<pair<int, double> > dirichlet_nodes;
  imeshH->get_property("dirichlet", dirichlet_nodes);
  vector<pair<string, Tensor> > conds;
  imeshH->get_property("conductivity_table", conds);
  int have_units = 0;
  string units;
  if (imeshH->mesh()->get_property("units", units)) have_units=1;

  for (int ii=0; ii<conds.size(); ii++) {
//    cerr << "New conds ["<<ii<<"] = "<<(*newConds)[ii].first<<" , "<<(*newConds)[ii].second<<"\n";
  }

  port_range_type range = get_iports("Electrodes");
  if (range.first != range.second) {
//    cerr << "Cloning data...\n";
    // get our own local copy of the Field and mesh

//    cerr << "Mesh pointer before = "<<imeshH->mesh().get_rep()<<"\n";
    imeshH.detach();
//    cerr << "Cloning mesh...\n";
    imeshH->mesh_detach();
//    cerr << "Done cloning!\n";
//    cerr << "Mesh pointer after = "<<imeshH->mesh().get_rep()<<"\n";

    TetVolField<int> *field = dynamic_cast<TetVolField<int>*>(imeshH.get_rep());
    TetVolMeshHandle mesh = field->get_typed_mesh();
    TetVolMeshHandle elecElemsH;
    port_map_type::iterator pi = range.first;
    while (pi != range.second) {
      FieldIPort *ielec = (FieldIPort *)get_iport(pi->second);
      if (!ielec->get(ielecH) || !ielecH.get_rep()) {
	++pi;
	continue;
      }

      CurveField<double> *elecFld = 
	dynamic_cast<CurveField<double>*>(ielecH.get_rep());
      if (!elecFld) {
	cerr << "InsertElectrodes: error - input electrode wasn't a CurveField<double>\n";
	++pi;
	continue;
      }
      double voltage = elecFld->fdata()[0];
      CurveMeshHandle elecMesh(elecFld->get_typed_mesh());
      Array1<Point> outer;
      Array1<Point> inner;
      CurveMesh::Node::iterator ni, ne;
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
      inner.add(inner[0]);
      outer.add(outer[0]);

      TetVolMesh* electrodeElements = 0;
      if (pi == range.first) electrodeElements = scinew TetVolMesh;

      // modify mergedDirichletNodes, and mergedMesh to have new nodes
      insertContourIntoTetMesh(dirichlet_nodes, mesh,
			       inner, outer, voltage, electrodeElements);
      
      if (pi == range.first) elecElemsH = electrodeElements;
      ++pi;
    }
    imeshH->set_property("dirichlet", dirichlet_nodes, false);
    imeshH->set_property("conductivity_table", conds, false);
    if (have_units) imeshH->mesh()->set_property("units", units, false);
    omesh->send(imeshH);
    if (elecElemsH.get_rep()) {
      TetVolField<double>* elec = scinew TetVolField<double>(elecElemsH, Field::NODE);
      FieldHandle elecH(elec);
      oelec->send(elecH);
    }
  } else {
    cerr << "Error - unable to get electrode.\n";
    omesh->send(imeshH);
  }
}
} // End namespace BioPSE
