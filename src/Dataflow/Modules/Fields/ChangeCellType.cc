/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  ChangeCellType: LatticeVol to TetVol - break hexes into tets
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/Array3.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Tester/RigorousTest.h>

#include <iostream>
using std::endl;
#include <stdio.h>



namespace SCIRun {

class ChangeCellType : public Module {
  FieldIPort* input_;
  FieldOPort* output_;
public:
  ChangeCellType(const string& id);
  virtual ~ChangeCellType();
  virtual void execute();
  
  template <class Data>
  void fill_tet_vol(LatticeVol<Data> *lvol, TetVol<Data> *tvol);
  
  template <class Iter, class Data>
  void set_data(const LatticeVol<Data> &src, TetVol<Data> &dst, 
		Iter begin, Iter end);
};

extern "C" Module* make_ChangeCellType(const string& id)
{
    return new ChangeCellType(id);
}

ChangeCellType::ChangeCellType(const string& id)
  : Module("ChangeCellType", id, Filter)
{
  input_ = new FieldIPort(this, "LatticeVol-in", FieldIPort::Atomic);
  add_iport(input_);
  output_ = new FieldOPort(this, "TetVol-out", FieldIPort::Atomic);
  add_oport(output_);
}

ChangeCellType::~ChangeCellType() {
}



// we assume that the min and max of the field are at the middle of the first
// and last node respectively.
// for a field with nx=ny=nz=3 and min=(0,0,0), max=(1,1,1), the "corners"
// of the cells are at x, y, z positions: -0.25, 0.25, 0.75, 1.25
// note: this is consistent with the SegFldToSurfTree and CStoSFRG modules

template <class Data>
void ChangeCellType::fill_tet_vol(LatticeVol<Data> *lvol, TetVol<Data> *tvol)
 {
  MeshHandle meshb = tet_vol_fh->get_mesh();
  
  MeshTet *mesh = dynamic_cast<MeshTet*>(meshb.get_rep());
  if (mesh == 0) return;

  int nx, ny, nz;
  nx=sf->nx;
  ny=sf->ny;
  nz=sf->nz;
  int nodes_size = (nx+1) * (ny+1) * (nz+1);
  vector<Point> *nodes = new vector<Point>(nodes_size);

  int offset=0;
  double dmin, dmax;
  sf->get_minmax(dmin, dmax);
  if (dmin==48 && dmax<54 && (sf->getRGChar()||sf->getRGUchar())) offset=48;
  
  Point min, max;
  sf->get_bounds(min,max);
  Vector d(max-min);
  d.x(d.x()/(2.*(nx-1.)));
  d.y(d.y()/(2.*(ny-1.)));
  d.z(d.z()/(2.*(nz-1.)));
  min-=d;
  max+=d;
  Array3<int> node_idx(nx+1, ny+1, nz+1);
  int currIdx=0;
  int i, j, k;

  for (i=0; i<nx+1; i++) {
    for (j=0; j<ny+1; j++) {
      for (k=0; k<nz+1; k++) {
	node_idx(i,j,k) = currIdx++;
	const Point p = min + Vector(d.x()*i, d.y()*j, d.z()*k);
	nodes.push_back(p);
      }
    }
  }

  // tets are 4 indecies into the node vector.
  vector<int> *tets = new vector<int>(4 * nodes_size);
  Array1<int> c(8); // each hex cell
  for (i=0; i<nx; i++) {
    for (int j=0; j<ny; j++) {
      for (int k=0; k<nz; k++) {

	c[0]=node_idx(i,j,k);
	c[1]=node_idx(i+1,j,k);
	c[2]=node_idx(i+1,j+1,k);
	c[3]=node_idx(i,j+1,k);
	c[4]=node_idx(i,j,k+1);
	c[5]=node_idx(i+1,j,k+1);
	c[6]=node_idx(i+1,j+1,k+1);
	c[7]=node_idx(i,j+1,k+1);
	if ((i+j+k)%2) {
	  // add in the tets
	  //e[0]=new Element(mesh, c[0], c[1], c[2], c[5]);	
	  tets.push_back(c[0]);
	  tets.push_back(c[1]);
	  tets.push_back(c[2]);
	  tets.push_back(c[5]);
	  
	  //e[1]=new Element(mesh, c[0], c[2], c[3], c[7]);
	  tets.push_back(c[0]);
	  tets.push_back(c[2]);
	  tets.push_back(c[3]);
	  tets.push_back(c[7]);

	  //e[2]=new Element(mesh, c[0], c[2], c[5], c[7]);
	  tets.push_back(c[0]);
	  tets.push_back(c[2]);
	  tets.push_back(c[5]);
	  tets.push_back(c[7]);

	  //e[3]=new Element(mesh, c[0], c[4], c[5], c[7]);
	  tets.push_back(c[0]);
	  tets.push_back(c[4]);
	  tets.push_back(c[5]);
	  tets.push_back(c[7]);
	  
	  //e[4]=new Element(mesh, c[2], c[5], c[6], c[7]);
	  tets.push_back(c[2]);
	  tets.push_back(c[5]);
	  tets.push_back(c[6]);
	  tets.push_back(c[7]);
	} else {
	  //e[0]=new Element(mesh, c[1], c[0], c[3], c[4]);
	  tets.push_back(c[1]);
	  tets.push_back(c[0]);
	  tets.push_back(c[3]);
	  tets.push_back(c[4]);

	  //e[1]=new Element(mesh, c[1], c[3], c[2], c[6]);
	  tets.push_back(c[1]);
	  tets.push_back(c[3]);
	  tets.push_back(c[2]);
	  tets.push_back(c[6]);

	  //e[2]=new Element(mesh, c[1], c[3], c[4], c[6]);
	  tets.push_back(c[1]);
	  tets.push_back(c[3]);
	  tets.push_back(c[4]);
	  tets.push_back(c[6]);

	  //e[3]=new Element(mesh, c[1], c[5], c[4], c[6]);
	  tets.push_back(c[1]);
	  tets.push_back(c[5]);
	  tets.push_back(c[4]);
	  tets.push_back(c[6]);

	  //e[4]=new Element(mesh, c[3], c[4], c[7], c[6]);
	  tets.push_back(c[3]);
	  tets.push_back(c[4]);
	  tets.push_back(c[7]);
	  tets.push_back(c[6]);
	}
      }
    }
  }
  
  // FIX_ME TODO:   tvol->compute_neighbors();
 
  // Load the data at the correct data location.
  switch (lvol->data_at()) {
  case Field::NODE :
    {
      set_data<MeshTet::Node::iterator>(lvol, tvol, 
				       tmesh->node_begin(),
				       tmesh->node_end());
    }
  break;
  case Field::EDGE:
    {
      set_data<MeshTet::Edge::iterator>(lvol, tvol, 
				       tmesh->edge_begin(),
				       tmesh->edge_end());
    }
    break;
  case Field::FACE:
    {
      set_data<MeshTet::Face::iterator>(lvol, tvol, 
				       tmesh->face_begin(),
				       tmesh->face_end());
    }
    break;
  case Field::CELL:
    {
      set_data<MeshTet::Cell::iterator>(lvol, tvol, 
				       tmesh->cell_begin(),
				       tmesh->cell_end());
    }
    break;
  }   
  
}

// Walk the data location, setting the data as we go.
// The Iter begin and end belong to the mesh in dst.
template <class Iter, class Data>
void
ChangeCellType::set_data(const LatticeVol<Data> &src, TetVol<Data> &dst, 
			 Iter begin, Iter end)
{
  Point p;
  LinearInterp ftor(8); // FIX_ME get size right...
  Iter iter = begin;
  while(iter != end)
  {
    MeshTetHandle mesh = dst.get_tet_mesh();
    p = mesh->get_center(*iter);
    // Always interps at data location matching Iter.
    src.interpolate(p, ftor); 
    dst[*iter] = ftor.result_;
  }
}

void 
ChangeCellType::execute()
{
  FieldHandle input_handle;
  update_state(NeedData);

  if (!input_->get(input_handle))
    return;
  if (!input_handle.get_rep()) {
    error("Empty field.");
    return;
  }
    

  string type_string = input_handle->type_name(0);
  if (type_string == "LatticeVol") {
    update_state(JustStarted);
      
    // then we have proper input
    if (input_handle->type_name(1) == "double")
    {
      TetVol<double> *tvol = new TetVol<double>();
      LatticeVol<double> *lvol = 
	dynamic_cast<LatticeVol<double>*>(input_handle.get_rep());
      fill_tet_vol<double>(*lvol, *tvol);
      output_->send(FieldHandle(tvol));
    }
    else if (input_handle->type_name(1) == "int")
    {
      TetVol<int> *tvol = new TetVol<int>();
      LatticeVol<int> *lvol = 
	dynamic_cast<LatticeVol<int>*>(input_handle.get_rep());
      fill_tet_vol<int>(*lvol, *tvol);
      output_->send(FieldHandle(tvol));
    }
    else if (input_handle->type_name(1) == "short")
    {
      TetVol<short> *tvol = new TetVol<short>();
      LatticeVol<short> *lvol = 
	dynamic_cast<LatticeVol<short>*>(input_handle.get_rep());
      fill_tet_vol<short>(*lvol, *tvol);
      output_->send(FieldHandle(tvol));
    }
    else if (input_handle->type_name(1) == "unsigned char")
    {
      TetVol<unsigned char> *tvol = new TetVol<unsigned char>();
      LatticeVol<unsigned char> *lvol = 
	dynamic_cast<LatticeVol<unsigned char>*>(input_handle.get_rep());
      fill_tet_vol<unsigned char>(*lvol, *tvol);
      output_->send(FieldHandle(tvol));
    } 
  } else {
    error("Field must be a LatticeVol.");
    return;
  }
}

} // End namespace SCIRun



