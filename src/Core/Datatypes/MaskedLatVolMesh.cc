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
 *  StructHexVolMesh.cc: Templated Mesh defined on a 3D Structured Grid
 *
 *  Written by:
 *   McKay Davis
 *   Department of Computer Science
 *   University of Utah
 *   Feb 2003
 *
 *  Copyright (C) 2002 SCI Group
 *
 */

/*
  See MaskedLatVolMesh.h for field/mesh comments.
*/

#include <Core/Datatypes/MaskedLatVolMesh.h>
#include <Core/Geometry/BBox.h>
#include <Core/Math/MusilRNG.h>
#include <iostream>


namespace SCIRun {

using namespace std;

PersistentTypeID MaskedLatVolMesh::type_id("MaskedLatVolMesh", "LatVolMesh", maker);


MaskedLatVolMesh::MaskedLatVolMesh():
  LatVolMesh(),
  synchronized_(0),
  nodes_(),
  masked_cells_(), 
  masked_nodes_count_(0),
  masked_edges_count_(0),
  masked_faces_count_(0)

{}

MaskedLatVolMesh::MaskedLatVolMesh(unsigned int x,
				   unsigned int y,
				   unsigned int z,
				   const Point &min,
				   const Point &max) :
  LatVolMesh(x, y, z, min, max),
  synchronized_(0),
  nodes_(),
  masked_cells_(), 
  masked_nodes_count_(0),
  masked_edges_count_(0),
  masked_faces_count_(0)
{}

MaskedLatVolMesh::MaskedLatVolMesh(const MaskedLatVolMesh &copy) :
  LatVolMesh(copy),
  synchronized_(copy.synchronized_),
  nodes_(copy.nodes_),
  masked_cells_(copy.masked_cells_), 
  masked_nodes_count_(copy.masked_nodes_count_),
  masked_edges_count_(copy.masked_edges_count_),
  masked_faces_count_(copy.masked_edges_count_)
{
}

void
MaskedLatVolMesh::begin(MaskedLatVolMesh::Node::iterator &itr) const
{
  itr = Node::iterator(this, min_i_, min_j_, min_k_);
  if (!check_valid(itr)) ++itr;
}

void
MaskedLatVolMesh::end(MaskedLatVolMesh::Node::iterator &itr) const
{
  itr = Node::iterator(this, min_i_, min_j_, min_k_ + nk_);
  //  if (!check_valid(itr)) { --itr; itr.next(); }  
}

void
MaskedLatVolMesh::size(MaskedLatVolMesh::Node::size_type &s) const
{
  s = Node::size_type(this,ni_,nj_,nk_);
}

void
MaskedLatVolMesh::begin(MaskedLatVolMesh::Cell::iterator &itr) const
{
  itr = Cell::iterator(this,  min_i_, min_j_, min_k_);
  if (!check_valid(itr)) ++itr;
}

void
MaskedLatVolMesh::end(MaskedLatVolMesh::Cell::iterator &itr) const
{
  itr = Cell::iterator(this, min_i_, min_j_, min_k_ + nk_-1);
  //  if (!check_valid(itr)) { --itr; itr.next(); }
}

void
MaskedLatVolMesh::size(MaskedLatVolMesh::Cell::size_type &s) const
{
  s = Cell::size_type(this,ni_-1, nj_-1,nk_-1);
}



void
MaskedLatVolMesh::begin(MaskedLatVolMesh::Edge::iterator &itr) const
{
  itr = Edge::iterator(this,min_i_,min_j_,min_k_,0);
  if (!check_valid(itr)) ++itr;
}

void
MaskedLatVolMesh::end(MaskedLatVolMesh::Edge::iterator &itr) const
{
  itr = Edge::iterator(this, min_i_, min_j_, min_k_,3);
  //  if (!check_valid(itr)) { --itr; itr.next(); }
}

void
MaskedLatVolMesh::size(MaskedLatVolMesh::Edge::size_type &s) const
{
  s = Edge::size_type(this,ni_,nj_,nk_);
}

void
MaskedLatVolMesh::begin(MaskedLatVolMesh::Face::iterator &itr) const
{
  itr = Face::iterator(this,min_i_,min_j_,min_k_,0);
  if (!check_valid(itr)) ++itr;
}

void
MaskedLatVolMesh::end(MaskedLatVolMesh::Face::iterator &itr) const
{
  itr = Face::iterator(this, min_i_, min_j_, min_k_,3);
  //  if (!check_valid(itr)) { --itr; itr.next(); }
}

void
MaskedLatVolMesh::size(MaskedLatVolMesh::Face::size_type &s) const
{
  s = Face::size_type(this,ni_,nj_,nk_);
}

//! get the child elements of the given index
void
MaskedLatVolMesh::get_nodes(Node::array_type &array, Edge::index_type e) const
{
  array.resize(2);
  array[0] = Node::index_type(this,e.i_,e.j_,e.k_);
  array[1] = Node::index_type(this,
			      e.i_ + (e.dir_ == 0 ? 1:0),
			      e.j_ + (e.dir_ == 1 ? 1:0),
			      e.k_ + (e.dir_ == 2 ? 1:0));
}

		           
void 
MaskedLatVolMesh::get_nodes(Node::array_type &array, Face::index_type f) const
{
  array.resize(4);
  array[0] = Node::index_type(this,f.i_,f.j_,f.k_);
  array[1] = Node::index_type(this,
			      f.i_ + (f.dir_ == 0 ? 1:0),
			      f.j_ + (f.dir_ == 1 ? 1:0),
			      f.k_ + (f.dir_ == 2 ? 1:0));
  array[2] = Node::index_type(this,
			      f.i_ + ((f.dir_ == 0 || f.dir_ == 2) ? 1:0),
			      f.j_ + ((f.dir_ == 0 || f.dir_ == 1) ? 1:0),
			      f.k_ + ((f.dir_ == 1 || f.dir_ == 2) ? 1:0));
  array[3] = Node::index_type(this,
			      f.i_ + (f.dir_ == 2 ? 1:0),
			      f.j_ + (f.dir_ == 0 ? 1:0),
			      f.k_ + (f.dir_ == 1 ? 1:0));

}

void 
MaskedLatVolMesh::get_nodes(Node::array_type &array, Cell::index_type idx) const
{
  array.resize(8);
  array[0].i_ = idx.i_;   array[0].j_ = idx.j_;   array[0].k_ = idx.k_;
  array[1].i_ = idx.i_+1; array[1].j_ = idx.j_;   array[1].k_ = idx.k_;
  array[2].i_ = idx.i_+1; array[2].j_ = idx.j_+1; array[2].k_ = idx.k_;
  array[3].i_ = idx.i_;   array[3].j_ = idx.j_+1; array[3].k_ = idx.k_;
  array[4].i_ = idx.i_;   array[4].j_ = idx.j_;   array[4].k_ = idx.k_+1;
  array[5].i_ = idx.i_+1; array[5].j_ = idx.j_;   array[5].k_ = idx.k_+1;
  array[6].i_ = idx.i_+1; array[6].j_ = idx.j_+1; array[6].k_ = idx.k_+1;
  array[7].i_ = idx.i_;   array[7].j_ = idx.j_+1; array[7].k_ = idx.k_+1;
}

void
MaskedLatVolMesh::get_edges(Edge::array_type &, Face::index_type) const
{}
void 
MaskedLatVolMesh::get_edges(Edge::array_type &, Cell::index_type) const
{}
void 
MaskedLatVolMesh::get_faces(Face::array_type &, Cell::index_type) const
{}


bool
MaskedLatVolMesh::check_valid(MaskedLatVolMesh::Node::index_type idx) const
{
  unsigned i = idx.i_, j = idx.j_, k = idx.k_;
  return (check_valid(i  ,j  ,k  ) ||
	  check_valid(i-1,j  ,k  ) ||
	  check_valid(i  ,j-1,k  ) ||
	  check_valid(i  ,j  ,k-1) ||
	  check_valid(i-1,j-1,k  ) ||
	  check_valid(i-1,j  ,k-1) ||
	  check_valid(i  ,j-1,k-1) ||
	  check_valid(i-1,j-1,k-1));	  
}


bool
MaskedLatVolMesh::check_valid(MaskedLatVolMesh::Edge::index_type idx) const
{

  bool val = false;
  if (idx.dir_ == 0)
  {
    val =  ((idx.i_ < min_i_ + ni_ - 1) &&
	     (check_valid(idx.i_  ,idx.j_  ,idx.k_  ) ||
	      check_valid(idx.i_  ,idx.j_-1,idx.k_  ) ||
	      check_valid(idx.i_  ,idx.j_  ,idx.k_-1) ||
	      check_valid(idx.i_  ,idx.j_-1,idx.k_-1)));
  }
  if (idx.dir_ == 1)
  {
    val =   ((idx.j_ < min_j_ + nj_ - 1) &&
	      (check_valid(idx.i_  ,idx.j_  ,idx.k_) ||
	       check_valid(idx.i_-1,idx.j_  ,idx.k_) ||
	       check_valid(idx.i_  ,idx.j_  ,idx.k_-1) ||
	       check_valid(idx.i_-1,idx.j_  ,idx.k_-1)));
  }
  if (idx.dir_ == 2)
  { 
    val =  ((idx.k_ < min_k_ + nk_ - 1) &&
	    (check_valid(idx.i_  ,idx.j_,  idx.k_) ||
	     check_valid(idx.i_-1,idx.j_,  idx.k_) ||
	     check_valid(idx.i_  ,idx.j_-1,idx.k_) ||
	     check_valid(idx.i_-1,idx.j_-1,idx.k_)));
  }
  return val;
}


bool
MaskedLatVolMesh::check_valid(MaskedLatVolMesh::Face::index_type idx) const
{
  if (idx.dir_ == 0)
  {
    return (idx.i_ < min_i_ + ni_ - 1 &&
	    idx.j_ < min_j_ + nj_ - 1 &&
	    (check_valid(idx.i_,idx.j_,idx.k_) ||
	     check_valid(idx.i_,idx.j_,idx.k_-1)));
  }
  if (idx.dir_ == 1)
  {
    return (idx.j_ < min_j_ + nj_ - 1 &&
	    idx.k_ < min_k_ + nk_ - 1 &&
	    (check_valid(idx.i_,idx.j_,idx.k_) ||
	     check_valid(idx.i_-1,idx.j_,idx.k_)));
  }
  if (idx.dir_ == 2)
  {
    return (idx.i_ < min_i_ + ni_ - 1 &&
	    idx.k_ < min_k_ + nk_ - 1 &&
	    (check_valid(idx.i_,idx.j_,idx.k_) ||
	     check_valid(idx.i_,idx.j_-1,idx.k_)));
  }

  return false;

}

bool
MaskedLatVolMesh::check_valid(MaskedLatVolMesh::Cell::index_type i) const
{
  return check_valid(i.i_,i.j_,i.k_);
}



bool
MaskedLatVolMesh::check_valid(MaskedLatVolMesh::Node::iterator i) const
{
  return check_valid(*i);
}

bool
MaskedLatVolMesh::check_valid(MaskedLatVolMesh::Edge::iterator i) const
{
  return check_valid(*i);
}

bool
MaskedLatVolMesh::check_valid(MaskedLatVolMesh::Face::iterator i) const
{
  return check_valid(*i);
}

bool
MaskedLatVolMesh::check_valid(MaskedLatVolMesh::Cell::iterator i) const
{
  return check_valid(i.i_,i.j_,i.k_);
}


//! This function updates the missing node, edge, and face count
//! when masking or unmasking one cell. 
//! Returns true if nodes, edges, or faces count is changed
bool
MaskedLatVolMesh::update_count(MaskedLatVolMesh::Cell::index_type c, 
			       bool masking)
{
  synchronized_ &= ~NODES_E;
  const bool i0 = (c.i_ > min_i_) && check_valid(c.i_-1, c.j_, c.k_);
  const bool j0 = (c.j_ > min_j_) && check_valid(c.i_, c.j_-1, c.k_);
  const bool k0 = (c.k_ > min_k_) && check_valid(c.i_, c.j_, c.k_-1);
  const bool i1 = (c.i_ < min_i_+ni_-1) && check_valid(c.i_+1, c.j_, c.k_);
  const bool j1 = (c.j_ < min_j_+nj_-1) && check_valid(c.i_, c.j_+1, c.k_);
  const bool k1 = (c.k_ < min_k_+nk_-1) && check_valid(c.i_, c.j_, c.k_+1);

  // These counts are the number of nodes, edges, faces that exist
  // ONLY from the presence of this cell, not because of the contribution 
  // of neighboring cells.
  const unsigned faces = (i0?0:1)+(i1?0:1)+(j0?0:1)+(j1?0:1)+(k0?0:1)+(k1?0:1);
  unsigned int   nodes = 0;
  unsigned int   edges = 0;

  if (faces == 6) {  
	nodes = 8; 
	edges = 12;
  } 
  else {
	if (faces == 5)	{ 
	  nodes = 4; edges = 8;
	}
	else { 
	  if (faces == 1 || faces == 0)	{ 
		nodes = 0; edges = 0; 
	  }
	  else { 
		if(faces == 4) {
		  if((i0 == i1) && (j0 == j1) && (k0 == k1)) {
			nodes = 0;
			edges = 4;
		  }
		  else {
			nodes = 2;
			edges = 5;
		  }
		}
		else {
		  if(faces == 3) {
			if((i0!=i1)&&(j0!=j1)&&(k0!=k1)) {
			  nodes = 1;
			  edges = 3;
			}
			else {
			  nodes = 0;
			  nodes = 2;
			}
		  }
		  else {
			if(faces == 2) {
			  if((i0 == i1) && (j0 == j1) && (k0 == k1)) {
				nodes = 0;
				edges = 0;
			  }
			  else {
				nodes = 0;
				edges = 1;
			  }
			}
		  }
		}
	  }
	}
  }

  // These nodes, edges, faces are being implicitly removed from the mesh
  // by the removal of this cell.
  if (masking)
  {
    masked_nodes_count_ += nodes;
    masked_edges_count_ += edges;
    masked_faces_count_ += faces;
  }
  // These ndoes, edges, & faces are being implicitly added back into the mesh
  // because this cell is being added back in
  else 
  {
    masked_nodes_count_ -= nodes;
    masked_edges_count_ -= edges;
    masked_faces_count_ -= faces;
  }

  return (faces == 0);
}

  


BBox
MaskedLatVolMesh::get_bounding_box() const
{
  // TODO:  return bounding box of valid cells only
  return LatVolMesh::get_bounding_box();
}


void
MaskedLatVolMesh::get_center(Point &result, const Node::index_type &idx) const
{
  ASSERT(check_valid(idx));
  LatVolMesh::get_center
    (result, LatVolMesh::Node::index_type(this,idx.i_,idx.j_,idx.k_));
}



void
MaskedLatVolMesh::get_center(Point &result, const Edge::index_type &idx) const
{
  ASSERT(check_valid(idx));
  LatVolMesh::get_center(result, LatVolMesh::Edge::index_type(unsigned(idx))); 
}

void
MaskedLatVolMesh::get_center(Point &result, const Face::index_type &idx) const
{
  ASSERT(check_valid(idx));
  LatVolMesh::get_center(result, LatVolMesh::Face::index_type(unsigned(idx))); 
}


void
MaskedLatVolMesh::get_center(Point &result, const Cell::index_type &idx) const
{
  ASSERT(check_valid(idx));
  LatVolMesh::get_center
    (result,LatVolMesh::Cell::index_type(this,idx.i_,idx.j_,idx.k_));
}


bool
MaskedLatVolMesh::locate(Node::index_type &idx, const Point &p)
{
  LatVolMesh::Node::index_type i(this,idx.i_,idx.j_,idx.k_);
  bool lat = LatVolMesh::locate(i,p);
  idx.i_ = i.i_; idx.j_ = i.j_; idx.k_ = i.k_; idx.mesh_ = this;
  if (lat && check_valid(idx))
  {
    return true;
  }
  else return false;    
}

bool
MaskedLatVolMesh::locate(Cell::index_type &idx, const Point &p)
{
  LatVolMesh::Cell::index_type i(this,idx.i_,idx.j_,idx.k_);
  bool lat = LatVolMesh::locate(i,p);
  idx.i_ = i.i_; idx.j_ = i.j_; idx.k_ = i.k_; idx.mesh_ = this;
  if (lat && check_valid(idx))
  {
    return true;
  }
  else return false;
}


void
MaskedLatVolMesh::get_weights(const Point &p,
			      Node::array_type &locs, vector<double> &w)
{
  const Point r = transform_.unproject(p);
  double ii = r.x();
  double jj = r.y();
  double kk = r.z();

  if (ii>(ni_-1) && (ii-(1.e-10))<(ni_-1)) ii=ni_-1-(1.e-10);
  if (jj>(nj_-1) && (jj-(1.e-10))<(nj_-1)) jj=nj_-1-(1.e-10);
  if (kk>(nk_-1) && (kk-(1.e-10))<(nk_-1)) kk=nk_-1-(1.e-10);
  if (ii<0 && ii>(-1.e-10)) ii=0;
  if (jj<0 && jj>(-1.e-10)) jj=0;
  if (kk<0 && kk>(-1.e-10)) kk=0;

  const unsigned int i = (unsigned int)floor(ii);
  const unsigned int j = (unsigned int)floor(jj);
  const unsigned int k = (unsigned int)floor(kk);

  if (i < (ni_-1) && i >= 0 &&
      j < (nj_-1) && j >= 0 &&
      k < (nk_-1) && k >= 0)
  {
    locs.resize(8);
    locs[0].i_ = i;   locs[0].j_ = j;   locs[0].k_ = k;   locs[0].mesh_=this;
    locs[1].i_ = i+1; locs[1].j_ = j;   locs[1].k_ = k;   locs[1].mesh_=this;
    locs[2].i_ = i+1; locs[2].j_ = j+1; locs[2].k_ = k;   locs[2].mesh_=this;
    locs[3].i_ = i;   locs[3].j_ = j+1; locs[3].k_ = k;   locs[3].mesh_=this;
    locs[4].i_ = i;   locs[4].j_ = j;   locs[4].k_ = k+1; locs[4].mesh_=this;
    locs[5].i_ = i+1; locs[5].j_ = j;   locs[5].k_ = k+1; locs[5].mesh_=this;
    locs[6].i_ = i+1; locs[6].j_ = j+1; locs[6].k_ = k+1; locs[6].mesh_=this;
    locs[7].i_ = i;   locs[7].j_ = j+1; locs[7].k_ = k+1; locs[7].mesh_=this;
    
    const double di = ii - (double)i;
    const double dj = jj - (double)j;
    const double dk = kk - (double)k;
    
    w.resize(8);
    w[0] = (1.0 - di) * (1.0 - dj) * (1.0 - dk);
    w[1] = di         * (1.0 - dj) * (1.0 - dk);
    w[2] = di         * dj         * (1.0 - dk);
    w[3] = (1.0 - di) * dj         * (1.0 - dk);
    w[4] = (1.0 - di) * (1.0 - dj) * dk;
    w[5] = di         * (1.0 - dj) * dk;
    w[6] = di         * dj         * dk;
    w[7] = (1.0 - di) * dj         * dk;
  }
}


void
MaskedLatVolMesh::get_weights(const Point &p,
			      Cell::array_type &l, vector<double> &w)
{
  Cell::index_type idx;
  if (locate(idx, p))
  {
    l.push_back(idx);
    w.push_back(1.0);
  }
}


double
MaskedLatVolMesh::get_size(Node::index_type idx) const
{
  ASSERT(check_valid(idx));
  LatVolMesh::Node::index_type i(this,idx.i_,idx.j_,idx.k_);
  return LatVolMesh::get_size(i);
}

double
MaskedLatVolMesh::get_size(Edge::index_type idx) const
{
  ASSERT(check_valid(idx));
  return LatVolMesh::get_size(LatVolMesh::Edge::index_type(unsigned(idx)));
}

double
MaskedLatVolMesh::get_size(Face::index_type idx) const
{
  ASSERT(check_valid(idx));
  return LatVolMesh::get_size(LatVolMesh::Face::index_type(unsigned(idx)));
}

double
MaskedLatVolMesh::get_size(Cell::index_type idx) const
{
  ASSERT(check_valid(idx));
  LatVolMesh::Cell::index_type i(this,idx.i_,idx.j_,idx.k_);
  return LatVolMesh::get_size(i);
}

void
MaskedLatVolMesh::mask_cell(Cell::index_type idx)
{
  update_count(idx,true);
  masked_cells_.insert(unsigned(idx));
}

void
MaskedLatVolMesh::unmask_cell(Cell::index_type idx)
{
  update_count(idx,false);
  masked_cells_.erase(unsigned(idx));
}


unsigned
MaskedLatVolMesh::num_masked_nodes() const
{
  return masked_nodes_count_;
}

unsigned
MaskedLatVolMesh::num_masked_edges() const
{
  return masked_edges_count_;
}

unsigned
MaskedLatVolMesh::num_masked_faces() const
{
  return masked_faces_count_;
}

unsigned
MaskedLatVolMesh::num_masked_cells() const
{
  return masked_cells_.size();
}

void 
MaskedLatVolMesh::
get_neighbors_stencil(vector<pair<bool,Cell::index_type> > &nbrs, 
					  Cell::index_type idx) const
{
  nbrs.clear();
  for (int k = idx.k_ - 1; k <= int(idx.k_ + 1); k++)
    for (int j = idx.j_ - 1; j <= int(idx.j_ + 1); j++)
      for (int i = idx.i_ - 1; i <= int(idx.i_ + 1); i++)
		if (i != int(idx.i_) || j != int(idx.j_) || k != int(idx.k_))
		  if (i >= int(min_i_) && j >= int(min_j_) && k >= int(min_k_) &&
			  i <= int(min_i_+ni_)-1 && j <= int(min_j_+nj_)-1 && 
			  i <= int(min_k_+nk_)-1 && check_valid(i,j,k))
			nbrs.push_back(make_pair(true,Cell::index_type(this,i,j,k)));
		  else
			nbrs.push_back(make_pair(false,Cell::index_type(0,0,0,0)));
}


void 
MaskedLatVolMesh::
get_neighbors_stencil(vector<pair<bool,Node::index_type> > &nbrs, 
		      Node::index_type idx) const
{
  nbrs.clear();
  for (int k = idx.k_ - 1; k <= int(idx.k_) + 1; k++)
    for (int j = idx.j_ - 1; j <= int(idx.j_) + 1; j++)
      for (int i = idx.i_ - 1; i <= int(idx.i_) + 1; i++)
	if (i != int(idx.i_) || j != int(idx.j_) || k != int(idx.k_))
	  if (i >= int(min_i_) && j >= int(min_j_) && k >= int(min_k_) &&
	      i <= int(min_i_+ni_) && j <= int(min_j_+nj_) &&
	      i <= int(min_k_+nk_) &&
	      check_valid(Node::index_type(this,i,j,k)))
	    nbrs.push_back(make_pair(true,Node::index_type(this,i,j,k)));
	  else
	    nbrs.push_back(make_pair(false,Node::index_type(0,0,0,0)));
}


void 
MaskedLatVolMesh::
get_neighbors_stencil(vector<pair<bool,Cell::index_type> > &nbrs, 
					  Node::index_type idx) const
{
  nbrs.clear();
  for (int k = idx.k_ - 1; k <= int(idx.k_); k++)
    for (int j = idx.j_ - 1; j <= int(idx.j_); j++)
      for (int i = idx.i_ - 1; i <= int(idx.i_); i++)
		if (i >= int(min_i_) && j >= int(min_j_) && k >= int(min_k_) &&
			i <= int(min_i_+ni_)-1 && j <= int(min_j_+nj_)-1 &&
			i <= int(min_k_+nk_)-1 && check_valid(i,j,k))
		  nbrs.push_back(make_pair(true,Cell::index_type(this,i,j,k)));
		else
		  nbrs.push_back(make_pair(false,Cell::index_type(0,0,0,0)));
}

    
unsigned int
MaskedLatVolMesh::get_sequential_node_index(const Node::index_type idx)
{
  if (!(synchronized_ & NODES_E))
  {
    nodes_.clear();
    int i = 0;
    Node::iterator node, nend;
    begin(node);
    end(nend);
    while (node != nend)
    {
      nodes_[*node] = i++;
      ++node;
    }
    synchronized_ |= NODES_E;
  }

  return nodes_[idx];
}


#define MASKED_LAT_VOL_MESH_VERSION 1

void
MaskedLatVolMesh::io(Piostream& stream)
{
  stream.begin_class(type_name(-1), MASKED_LAT_VOL_MESH_VERSION);

  LatVolMesh::io(stream);

  // IO data members, in order
  vector<unsigned> masked_vec(masked_cells_.begin(), 
			      masked_cells_.end());
  Pio(stream, masked_vec);
  if (stream.reading())
    {
      masked_cells_.clear();
      masked_cells_.insert(masked_vec.begin(), masked_vec.end());
    }

  stream.end_class();
}

const string
MaskedLatVolMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "MaskedLatVolMesh";
  return name;
}

const TypeDescription*
MaskedLatVolMesh::get_type_description() const
{
  return SCIRun::get_type_description((MaskedLatVolMesh *)0);
}

const TypeDescription*
get_type_description(MaskedLatVolMesh *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("MaskedLatVolMesh",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}


const TypeDescription* 
get_type_description(MaskedLatVolMesh::NodeIndex *)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("MaskedLatVolMesh::NodeIndex",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription* 
get_type_description(MaskedLatVolMesh::CellIndex *)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("MaskedLatVolMesh::CellIndex",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}


const TypeDescription*
get_type_description(MaskedLatVolMesh::Node *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("MaskedLatVolMesh::Node",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(MaskedLatVolMesh::Edge *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("MaskedLatVolMesh::Edge",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(MaskedLatVolMesh::Face *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("MaskedLatVolMesh::Face",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(MaskedLatVolMesh::Cell *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("MaskedLatVolMesh::Cell",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}


} // namespace SCIRun
