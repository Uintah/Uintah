/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

//    File   : RefineTetVol.cc
//    Author : Martin Cole
//    Date   : Thu Nov  6 16:35:41 2003

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Datatypes/TetVolField.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <sci_hash_map.h>
#include <algorithm>

namespace SCIRun {

using std::cerr;
using std::endl;

class RefineTetVol : public Module
{
public:
  RefineTetVol(GuiContext* ctx);
  virtual ~RefineTetVol();

  virtual void execute();

private:
  void delete_orphans(set<unsigned int> &removed_cells);
  void subdivide_to_level(unsigned);
  void locally_refine(TetVolField<char> *control);

  void subdivide(TetVolMesh::Cell::index_type ci, 
		 set<unsigned int, less<unsigned int> >&removed_cells);

  int                       input_generation_;
  TetVolField<char>        *subdivided_;
  vector<unsigned char>     levels_;
  GuiInt                    cell_index_;
  GuiString                 execution_mode_;
};


DECLARE_MAKER(RefineTetVol)
  
  RefineTetVol::RefineTetVol(GuiContext* ctx) : 
    Module("RefineTetVol", ctx, Filter, "FieldsCreate", "SCIRun"),
    input_generation_(-1),
    subdivided_(0),
    cell_index_(ctx->subVar("cell_index")),
    execution_mode_(ctx->subVar("execution_mode"))
{
  
}


RefineTetVol::~RefineTetVol()
{
}

void
RefineTetVol::execute() 
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input TetVol");
  FieldHandle ifieldhandle;

  FieldIPort *control_port = (FieldIPort *)get_iport("Refinement Control");
  FieldHandle control_handle;
  
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep())) {
    error("RefineTetvol must have an input field to work with");
    return;
  }
  MeshHandle mesh_handle = ifieldhandle->mesh();
  TetVolMesh *tvm = dynamic_cast<TetVolMesh*>(mesh_handle.get_rep());
  if (!tvm) {
    error("Must have input Field that has a TetVolMesh");
    error("Error: Invalid Input");
    return;
  }
  TetVolField<char> *control = 0;

  if (control_port->get(control_handle) && control_handle.get_rep()) {
    remark("Attempting to use Field in Refinement Control port to control refinement.");
    
    // Make sure it is the correct type of field.
    control = dynamic_cast<TetVolField<char>*>(control_handle.get_rep());
    if (!control) {
      error("second input must be a TetVolField<char>");
      return;
    }      
    MeshHandle mesh_handle = control_handle->mesh();
    TetVolMesh *c_mesh = dynamic_cast<TetVolMesh*>(mesh_handle.get_rep());
    
    // make sure it is the same mesh as input port 1.
    if (tvm != c_mesh) {
      error("second input mesh and first must be a shared mesh.");
      control = 0;
      return;
    }
  }

  // If we have a new input start over with that input.
  if (input_generation_ != ifieldhandle->generation) {
    input_generation_ = ifieldhandle->generation;
    // About to subdivide the input so detach.
    ifieldhandle.detach();
    if (subdivided_) delete subdivided_;
    
    TetVolField<char>::mesh_handle_type mh(tvm);
    subdivided_ = scinew TetVolField<char>(mh, 0);
    // About to create new geometry so detach the mesh.
    subdivided_->mesh_detach(); 

    TetVolMesh::Cell::size_type num_tets;
    mh->size(num_tets);    
    subdivided_->mesh()->synchronize(Mesh::FACE_NEIGHBORS_E | 
				     Mesh::EDGE_NEIGHBORS_E );

    // All start with 0th level subdivision.
    levels_.clear();
    levels_.resize(num_tets);
  }

  if (control) {
      
    locally_refine(control);

  } else {
    execution_mode_.reset();
    cell_index_.reset();
    if (execution_mode_.get() == "sub_one" && cell_index_.get() != -1) {
      // Could delete cells so cache those indices.
      set<unsigned int, less<unsigned int> > rem;
      subdivide(cell_index_.get(), rem);
      delete_orphans(rem);
    } else if (cell_index_.get() != -1) {
      subdivide_to_level(cell_index_.get());
    }
  }
  FieldOPort *out_port = (FieldOPort *)get_oport("Refined TetVol");
  // always send a copy of subdivided_.  bad things could happen if another 
  // module is iterating over this field while this module is changing it.

  TetVolField<char> *out = subdivided_->clone();
  out->mesh_detach();
  // copy the levels to the data
  out->resize_fdata();
  TetVolField<char>::mesh_handle_type out_mesh = out->get_typed_mesh();
  TetVolMesh::Cell::iterator ci, c_end;
  out_mesh->begin(ci);
  out_mesh->end(c_end);
  while(ci != c_end) {
    TetVolMesh::Cell::index_type cidx = *ci;
    ++ci;
    out->set_value(levels_[cidx], cidx);
  }
  
  out_port->send(FieldHandle(out));
}

void 
RefineTetVol::delete_orphans(set<unsigned int> &rem)
{
  if (rem.empty()) return;
  // Delete the largest index first.
  set<unsigned int, less<unsigned int> >::reverse_iterator iter = 
    rem.rbegin();
  while (iter != rem.rend()) {
    // clean up levels list
    TetVolMesh::Cell::index_type ci = *iter++;
    vector<unsigned char>::iterator liter = levels_.begin() + ci;
    levels_.erase(liter);
  }
  TetVolField<char>::mesh_handle_type mh = subdivided_->get_typed_mesh();
  mh->delete_cells(rem);
}

inline 
bool
is_even(unsigned v) {
  return ! (v % 2);
}

// recursive call..
void 
RefineTetVol::subdivide(TetVolMesh::Cell::index_type ci, 
			set<unsigned int, less<unsigned int> > &removed_cells) 
{
  if (! subdivided_) return;
  if (removed_cells.count(ci)) {
    //cerr << "dont subdivide a removed cell" << endl;
    return;
  }
  TetVolField<char>::mesh_handle_type mh = subdivided_->get_typed_mesh();
  TetVolMesh *mesh = mh.get_rep();

  if (is_even(levels_[ci])) {
    // do a center split
    Point center;
    mesh->get_center(center, ci);
    
    TetVolMesh::Node::index_type ni;
    TetVolMesh::Cell::array_type new_tets;
    if (! mesh->insert_node_in_cell(new_tets, ci, ni, center)) {
      cerr << "could not insert a node at the center of tet" << endl;
      return;
    }
    
    // incr level and check each new tet for swaps.
    TetVolMesh::Cell::size_type num_tets;
    mh->size(num_tets); 
    int lev = levels_[ci] + 1;
    levels_.resize(num_tets);
    TetVolMesh::Cell::array_type::iterator citer = new_tets.begin();
    while (citer != new_tets.end()) {
      if (removed_cells.count(*citer)) {
	cerr << "skipping removed index" << endl;
	++citer;
	continue;
      }
      TetVolMesh::Cell::index_type ind = *citer;
      ++citer;
      // fix index level
      levels_[ind] = lev;
      // and check for nbor face swaps.
      // always check across face that does not contain the just added point.
      TetVolMesh::Face::index_type fi;
      if (! mesh->get_face_opposite_node(fi, ind, ni)) {
	error("Bad input to get_face_opposite_node");
	return;
      }
      TetVolMesh::Cell::index_type nbor_ci;
      if (mesh->get_neighbor(nbor_ci, ind, fi)) {     
	// if nbor is the same level then we swap.
	if (levels_[ind] == levels_[nbor_ci]) {
	  // The swap takes the 5 points that make up 2 tets who share a face,
	  // and connects the 2 extreme points through the shared face, 
	  // eliminating that face, and creating 3 new tets.
	  TetVolMesh::Cell::array_type split_tets(3);
	  TetVolMesh::Node::index_type n1, n2;
	  if (!mesh->split_2_to_3(split_tets, n1, n2, ind, nbor_ci, fi)) {
	    error("could not split 2 to 3");
	    return;
	  }
	  unsigned lev = levels_[ind] + 1;
	  levels_.resize(levels_.size() + 1);
	  levels_[split_tets[0]] = lev;
	  levels_[split_tets[1]] = lev;
	  levels_[split_tets[2]] = lev;

	  // If we are now a lev % 4 check for a 3 to 2 combine
	  if (lev % 4 == 0) {
	    // find the 3 tets around the edge that is not connected to the 
	    // edge these three tets are connected to..

	    TetVolMesh::Cell::array_type::iterator siter = split_tets.begin();
	    while (siter != split_tets.end()) {
	      if (removed_cells.count(*siter)) {
		cerr << "skipping removed index" << endl;
		++siter;
		continue;
	      }
	      TetVolMesh::Edge::index_type ei;
	      ASSERT(mesh->get_edge(ei, *siter, n1, n2));

	      TetVolMesh::Edge::index_type opp;
	      opp = TetVolMesh::Edge::opposite_edge(ei);

	      TetVolMesh::Cell::array_type connected_cells;
	      mesh->get_cells(connected_cells, opp);



	      if (connected_cells.size() == 3 &&
		  levels_[connected_cells[0]] == lev &&
		  levels_[connected_cells[1]] == lev &&
		  levels_[connected_cells[2]] == lev) {

		// all same level and are a multiple of level 4
		// combine 3 tets into 2
		TetVolMesh::Cell::index_type removed;
		mesh->combine_3_to_2(removed, opp);
		// the levels of the 2 new tets remain at lev		
		removed_cells.insert(removed);
	      }
	      ++siter;
	    }
	  }
	}
      }
    } 
  } else {
    // check the nbor across the face from the previously added node.
    // this is always the same relative nbor due to way we do the split    

    // due to the way the mesh adds tets in a center split we know which 
    // face to look across.
    TetVolMesh::Face::index_type face = ci * 4 + 3;
    
    TetVolMesh::Cell::index_type nbor_ci;
    if (mesh->get_neighbor(nbor_ci, ci, face)) {     
      // if nbor is the same level then we swap.
      if (levels_[ci] == levels_[nbor_ci] - 2) {
	subdivide(nbor_ci, removed_cells);
	mesh->get_neighbor(nbor_ci, ci, face);
      }      
      subdivide(nbor_ci, removed_cells);
    } else {
      // boundary face, do face split.

      TetVolMesh::Cell::array_type split_tets(3);
      TetVolMesh::Node::index_type ni;
      int lev = levels_[ci] + 1;
      mesh->split_cell_at_boundary(split_tets, ni, ci, face);
      // single tet became 3.
      levels_.resize(levels_.size() + 2);
      levels_[split_tets[0]] = lev;
      levels_[split_tets[1]] = lev;
      levels_[split_tets[2]] = lev;
    }
  }
}

void
RefineTetVol::subdivide_to_level(unsigned)
{  
  const double max_vol = 0.125L;
  TetVolField<char>::mesh_handle_type mh = subdivided_->get_typed_mesh();
  set<unsigned int> removed;
  bool done = false;
  update_progress(0.0);
  while(! done) {
    done = true;
    TetVolMesh::Cell::size_type num_tets;
    mh->size(num_tets);

    for (unsigned int i = 0; i < num_tets; i++) {
      if (removed.count(i)) {
	continue;
      }
      update_progress((double)i / (double)num_tets);
      double cur_vol = mh->get_volume(i);
      
      while (cur_vol < max_vol) {
	done = false;
	subdivide(i, removed);
	if (removed.count(i)) {
	  break;
	}
	cur_vol = mh->get_volume(i);
      }
    }
  }
  delete_orphans(removed);
  update_progress(1.0);
}

#if defined(ORIGINAL_WORKING_CODE)
void
RefineTetVol::subdivide_to_level(unsigned target_level)
{  
  TetVolField<char>::mesh_handle_type mh = subdivided_->get_typed_mesh();
  set<unsigned int> removed;
  bool done = false;
  while(! done) {
    done = true;
    TetVolMesh::Cell::size_type num_tets;
    mh->size(num_tets);

    for (unsigned int i = 0; i < num_tets; i++) {
      if (removed.count(i)) {
	continue;
      }
      while (levels_[i] < target_level) {
	done = false;
	subdivide(i, removed);
      }
    }
  }
  delete_orphans(removed);
}
#endif
void
RefineTetVol::locally_refine(TetVolField<char> *control)
{  
  TetVolField<char>::mesh_handle_type cnt = control->get_typed_mesh();
  TetVolField<char>::mesh_handle_type mh = subdivided_->get_typed_mesh();
  set<unsigned int> removed;
  
  TetVolMesh::Cell::size_type num_tets;
  cnt->size(num_tets);

  for (unsigned int i = 0; i < num_tets; i++) {
    if (removed.count(i)) {
      continue;
    }
    unsigned int val = control->value((TetVolMesh::Cell::index_type)i);
    Point center;
    cnt->get_center(center, (TetVolMesh::Cell::index_type)i);
    TetVolMesh::Cell::index_type ci;
    mh->locate(ci, center);
    
    if (val == 0) continue;
    if (val > 0) {
      // indicates subdivision.
      while (levels_[ci] < val) {
	subdivide(ci, removed);
      }
    } else {
      //FIX_ME implement simplification.
      static int err = 0;
      if (!err) {
	warning("simplification not yet implemented, continuing...");
	++err;
	continue;
      }
    }
  }
  delete_orphans(removed);
}

} // end namespace SCIRun

