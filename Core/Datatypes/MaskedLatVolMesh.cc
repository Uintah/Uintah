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

PersistentTypeID MaskedLatVolMesh::type_id("MaskedLatVolMesh", "Mesh", maker);


MaskedLatVolMesh::MaskedLatVolMesh():
  masked_cells_(), 
  masked_nodes_count_(0),
  masked_edges_count_(0),
  masked_faces_count_(0)

{}

MaskedLatVolMesh::MaskedLatVolMesh(unsigned int x,
				   unsigned int y,
				   unsigned int z) :
  LatVolMesh(x, y, z, Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0)),
  masked_cells_(), 
  masked_nodes_count_(0),
  masked_edges_count_(0),
  masked_faces_count_(0)
{}

MaskedLatVolMesh::MaskedLatVolMesh(const MaskedLatVolMesh &copy) :
  LatVolMesh(copy),
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
  if (!check_valid(itr)) --itr;
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
  if (!check_valid(itr)) --itr;
}

void
MaskedLatVolMesh::size(MaskedLatVolMesh::Cell::size_type &s) const
{
  s = Cell::size_type(this,ni_-1, nj_-1,nk_-1);
}



void
MaskedLatVolMesh::begin(MaskedLatVolMesh::Edge::iterator &itr) const
{
  ASSERTFAIL("Not Finished");
  itr = Edge::iterator(0);
}

void
MaskedLatVolMesh::end(MaskedLatVolMesh::Edge::iterator &itr) const
{
  ASSERTFAIL("Not Finished");
  itr = ((ni_-1) * nj_ * nk_) + (ni_ * (nj_-1) * nk_) + (ni_ * nj_ * (nk_-1));
}

void
MaskedLatVolMesh::size(MaskedLatVolMesh::Edge::size_type &s) const
{
  s = (((ni_-1) * nj_ * nk_) + 
       (ni_ * (nj_-1) * nk_) + 
       (ni_ * nj_ * (nk_-1)) -
       masked_edges_count_);
}

void
MaskedLatVolMesh::begin(MaskedLatVolMesh::Face::iterator &itr) const
{
  ASSERTFAIL("Not Finished");
  itr = Face::iterator(0);
}

void
MaskedLatVolMesh::end(MaskedLatVolMesh::Face::iterator &itr) const
{
  ASSERTFAIL("Not Finished");
  itr = (ni_-1) * (nj_-1) * nk_ +
    ni_ * (nj_ - 1 ) * (nk_ - 1) +
    (ni_ - 1) * nj_ * (nk_ - 1);
}

void
MaskedLatVolMesh::size(MaskedLatVolMesh::Face::size_type &s) const
{
  s =  (((ni_-1) * (nj_-1) * nk_) +
	(ni_ * (nj_ - 1 ) * (nk_ - 1)) +
	((ni_ - 1) * nj_ * (nk_ - 1)) -
	masked_faces_count_);
}



bool
MaskedLatVolMesh::check_valid(MaskedLatVolMesh::Node::index_type idx) const
{
  // TODO: this still isnt quite done, need to check if node is on max boundary

  unsigned i = idx.i_, j = idx.j_, k = idx.k_;
  if (check_valid(i,j,k)) return true;
  else if (i != min_i_ && check_valid(i-1,j,k)) return true;
  else if (j != min_j_ && check_valid(i,j-1,k)) return true;
  else if (i != min_i_ && j != min_j_ && check_valid(i-1,j-1,k))
    return true;
  else if (k != min_k_)
  {
    k--;
    if (check_valid(i,j,k)) return true;
    else if (i != min_i_ && check_valid(i-1,j,k)) return true;
    else if (j != min_j_ && check_valid(i,j-1,k)) return true;
    else if (i != min_i_ && j != min_j_ && check_valid(i-1,j-1,k))
      return true;
  }
  return false;
}


void
MaskedLatVolMesh::convert_edge_indexing(MaskedLatVolMesh::Edge::index_type idx,
					unsigned &dir, unsigned &i, 
					unsigned &j, unsigned &k) const
{
  const unsigned int j_start= (ni_-1)*nj_*nk_; 
  const unsigned int k_start = ni_*(nj_-1)*nk_ + j_start; 
  
  if (idx < j_start) {
    dir = 0;
    i = idx % (ni_ - 1);
    const unsigned jk = idx / (ni_ - 1);
    j = jk % nj_;
    k = jk / nj_;
    
  } else if (idx < k_start) {
    dir = 1;
    idx.index_ -= j_start;
    j = idx % (nj_ - 1);
    const unsigned ik = idx / (nj_ - 1);
    i = ik / nk_;
    k = ik % nk_;
    
  } else {
    dir = 2;
    idx.index_ -= k_start;
    k = idx % (nk_ - 1);
    const unsigned ij = idx / (nk_ - 1);
    i = ij % ni_;
    j = ij / ni_;    
  }
}


bool
MaskedLatVolMesh::check_valid(MaskedLatVolMesh::Edge::index_type idx) const
{
  unsigned i,j,k,dir;
  convert_edge_indexing (idx, dir, i, j, k);
  if (dir == 0)
  {
    return (check_valid(Cell::index_type(this,i,j,k)) ||
	    j-1 >= min_j_ && check_valid(Cell::index_type(this,i,j-1,k)) ||
	    k-1 >= min_k_ && check_valid(Cell::index_type(this,i,j,k-1)) ||
	    j-1 >= min_j_ && k-1 >= min_k_ && 
	    check_valid(Cell::index_type(this,i,j-1,k-1)));
  }
  if (dir == 1)
  {
    return (check_valid(Cell::index_type(this,i,j,k)) ||
	    i-1 >= min_i_ && check_valid(Cell::index_type(this,i-1,j,k)) ||
	    k-1 >= min_k_ && check_valid(Cell::index_type(this,i,j,k-1)) ||
	    i-1 >= min_i_ && k-1 >= min_k_ && 
	    check_valid(Cell::index_type(this,i-1,j,k-1)));
  }
  if (dir == 2)
  {
    return (check_valid(Cell::index_type(this,i,j,k)) ||
	    i-1 >= min_i_ && check_valid(Cell::index_type(this,i-1,j,k)) ||
	    j-1 >= min_j_ && check_valid(Cell::index_type(this,i,j-1,k)) ||
	    i-1 >= min_i_ && j-1 >= min_j_ && 
	    check_valid(Cell::index_type(this,i-1,j-1,k)));
  }

  return true;
}


bool
MaskedLatVolMesh::check_valid(MaskedLatVolMesh::Face::index_type idx) const
{
  unsigned i,j,k,dir;
#ifdef TODO
  convert_face_indexing (idx, dir, i, j, k);
#endif
  if (dir == 0)
  {
    return (check_valid(Cell::index_type(this,i,j,k)) ||
	    k-1 >= min_k_ && check_valid(Cell::index_type(this,i,j,k-1)));
  }
  if (dir == 1)
  {
    return (check_valid(Cell::index_type(this,i,j,k)) ||
	    i-1 >= min_i_ && check_valid(Cell::index_type(this,i-1,j,k)));
  }
  if (dir == 2)
  {
    return (check_valid(Cell::index_type(this,i,j,k)) ||
	    j-1 >= min_j_ && check_valid(Cell::index_type(this,i,j-1,k)));
  }

  return true;

}

bool
MaskedLatVolMesh::check_valid(MaskedLatVolMesh::Cell::index_type i) const
{
  return masked_cells_.find(unsigned(i)) != masked_cells_.end();
}



bool
MaskedLatVolMesh::check_valid(MaskedLatVolMesh::Node::iterator i) const
{
  return check_valid(Node::index_type(this, i.i_, i.j_, i.k_));
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



//! Returns true if nodes, edges, or faces count is changed
bool
MaskedLatVolMesh::update_count(MaskedLatVolMesh::Cell::index_type c, 
			       bool masking)
{
  const bool i0 = (c.i_ > min_i_) && check_valid(c.i_-1, c.j_, c.k_);
  const bool j0 = (c.j_ > min_j_) && check_valid(c.i_, c.j_-1, c.k_);
  const bool k0 = (c.k_ > min_k_) && check_valid(c.i_, c.j_, c.k_-1);
  const bool i1 = (c.i_ < min_i_+ni_) && check_valid(c.i_+1, c.j_, c.k_);
  const bool j1 = (c.j_ < min_j_+nj_) && check_valid(c.i_, c.j_+1, c.k_);
  const bool k1 = (c.k_ < min_k_+nk_) && check_valid(c.i_, c.j_, c.k_+1);
  const unsigned faces = (i0?1:0)+(i1?1:0)+(j0?1:0)+(j1?1:0)+(k0?1:0)+(k1?1:0);
  unsigned nodes, edges;

  if (faces == 0) { nodes = 8; edges = 12; } 
  if (faces == 1) { nodes = 4; edges = 8; }
  if (faces == 5 || faces == 6) { nodes = 0; edges = 0; }
  
  const bool is = i0 == i1;
  const bool js = j0 == j1;
  const bool ks = k0 == k1;
  
  if (faces == 2)
    if (!is || !js || !ks)
      { nodes = 2; edges = 5; }		// 2 faces sharing an edge
    else
      { nodes = 0; edges = 4; }		// 2 faces on opposite sides

  if (faces == 3)
    if (is || js || ks)			
      { nodes = 0; edges = 2; }		// 3 faces not all sharing one node
    else
      { nodes = 1; edges = 3; }		// 3 faces all sharing one node
  
  if (faces == 4)
    if (!is || !js || !ks)
      { nodes = 0; edges = 1; }		// 4 faces, 2 missing share edge
    else
      { nodes = 0; edges = 0; }		// 4 faces, 2 missing on opposite sides

  if (masking)
  {
    masked_nodes_count_ += nodes;
    masked_edges_count_ += edges;
    masked_faces_count_ += 6-faces;
  }
  else
  {
    masked_nodes_count_ += nodes;
    masked_edges_count_ -= edges;
    masked_faces_count_ -= 6-faces;
  }

  return faces != 6;
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
  check_valid(idx);
  LatVolMesh::get_center
    (result, LatVolMesh::Node::index_type(this,idx.i_,idx.j_,idx.k_));
}



void
MaskedLatVolMesh::get_center(Point &result, const Edge::index_type &idx) const
{
  check_valid(idx);
  LatVolMesh::get_center (result, idx); 
}

void
MaskedLatVolMesh::get_center(Point &result, const Face::index_type &idx) const
{
  check_valid(idx);
  LatVolMesh::get_center(result, idx); 

}


void
MaskedLatVolMesh::get_center(Point &result, const Cell::index_type &idx) const
{
  check_valid(idx);
  LatVolMesh::get_center
    (result,LatVolMesh::Cell::index_type(this,idx.i_,idx.j_,idx.k_));
}


bool
MaskedLatVolMesh::locate(Node::index_type &idx, const Point &p)
{
  LatVolMesh::Node::index_type i(this,idx.i_,idx.j_,idx.k_);
  return (LatVolMesh::locate(i,p) && check_valid(idx));
}

bool
MaskedLatVolMesh::locate(Cell::index_type &idx, const Point &p)
{
  LatVolMesh::Cell::index_type i(this,idx.i_,idx.j_,idx.k_);
  return (LatVolMesh::locate(i,p) &&  check_valid(idx));
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
  return LatVolMesh::get_size(idx);
}

double
MaskedLatVolMesh::get_size(Face::index_type idx) const
{
  ASSERT(check_valid(idx));
  return LatVolMesh::get_size(idx);
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
  return 0;
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
		      Cell::index_type &idx) const
{
  nbrs.clear();
  for (int k = idx.k_ - 1; k <= int(idx.k_ + 1); k++)
    for (int j = idx.j_ - 1; j <= int(idx.j_ + 1); j++)
      for (int i = idx.i_ - 1; k <= int(idx.i_ + 1); i++)
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
		      Node::index_type &idx) const
{
  nbrs.clear();
  for (int k = idx.k_ - 1; k <= int(idx.k_) + 1; k++)
    for (int j = idx.j_ - 1; j <= int(idx.j_) + 1; j++)
      for (int i = idx.i_ - 1; k <= int(idx.i_) + 1; i++)
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
		      Node::index_type &idx) const
{
  nbrs.clear();
  for (int k = idx.k_ - 1; k <= int(idx.k_); k++)
    for (int j = idx.j_ - 1; j <= int(idx.j_); j++)
      for (int i = idx.i_ - 1; k <= int(idx.i_); i++)
	if (i >= int(min_i_) && j >= int(min_j_) && k >= int(min_k_) &&
	    i <= int(min_i_+ni_)-1 && j <= int(min_j_+nj_)-1 &&
	    i <= int(min_k_+nk_)-1 && check_valid(i,j,k))
	  nbrs.push_back(make_pair(true,Cell::index_type(this,i,j,k)));
	else
	  nbrs.push_back(make_pair(false,Cell::index_type(0,0,0,0)));
}

    

#define STRUCT_HEX_VOL_MESH_VERSION 1

void
MaskedLatVolMesh::io(Piostream& stream)
{
  stream.begin_class(type_name(-1), STRUCT_HEX_VOL_MESH_VERSION);

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

} // namespace SCIRun
