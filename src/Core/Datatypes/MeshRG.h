/*
 *  MeshRG.h: Templated Mesh defined on a 3D Regular Grid
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 *
 */

#ifndef SCI_project_MeshRG_h
#define SCI_project_MeshRG_h 1

#include <Core/Datatypes/Datatype.h>
#include <Core/Geometry/BBox.h>


namespace SCIRun {

class SCICORESHARE MeshRG : public Datatype
{

  struct IPoint
  {
    IPoint() {}
    IPoint(int i, int j, int k) : i_(i), j_(j), k_(k) {}

    int i_, j_, k_;
  };


  struct NCIter : public IPoint
  {
    NCIter(const MeshRG *m, int i, int j, int k) : IPoint(i, j, k), mesh_(m) {}

    const IPoint &operator *() { return *this; }

    bool operator ==(const NCIter &a)
    {
      return i_ == a.i_ && j_ == a.j_ && k_ == a.k_ && mesh_ == a.mesh_;
    }

    bool operator !=(const NCIter &a)
    {
      return !(*this == a);
    }

    const MeshRG *mesh_;
  };


  struct NodeIter : public NCIter
  {
    NodeIter(const MeshRG *m, int i, int j, int k) : NCIter(m, i, j, k) {}

    NodeIter &operator++()
    {
      i_++;
      if (i_ > mesh_->nx_)
      {
	i_ = 0;
	j_++;
	if (j_ > mesh_->ny_)
	{
	  j_ = 0;
	  k_++;
	}
      }
      return *this;
    }

    NodeIter operator++(int)
    {
      NodeIter result(*this);
      operator++();
      return result;
    }
  };


  struct CellIter : public NCIter
  {
    CellIter(const MeshRG *m, int i, int j, int k) : NCIter(m, i, j, k) {}

    CellIter &operator++()
    {
      i_++;
      if (i_ >= mesh_->nx_)
      {
	i_ = 0;
	j_++;
	if (j_ >= mesh_->ny_)
	{
	  j_ = 0;
	  k_++;
	}
      }
      return *this;
    }

    CellIter operator++(int)
    {
      CellIter result(*this);
      operator++();
      return result;
    }
  };

public:

  typedef IPoint          node_index;
  typedef NodeIter        node_iterator;

  typedef void *          edge_index; 
  typedef void *          edge_iterator;
	                
  typedef void *          face_index;
  typedef void *          face_iterator;
	                
  typedef IPoint          cell_index;
  typedef CellIter        cell_iterator;

  typedef vector<node_index> node_array;
  typedef vector<edge_index> edge_array;
  typedef vector<face_index> face_array;


  MeshRG(int x, int y, int z);
  MeshRG(const MeshRG &);
  virtual ~MeshRG();


  virtual BBox get_bounding_box() const;


  node_iterator node_begin() const;
  node_iterator node_end() const;
  edge_iterator edge_begin() const;
  edge_iterator edge_end() const;
  face_iterator face_begin() const;
  face_iterator face_end() const;
  cell_iterator cell_begin() const;
  cell_iterator cell_end() const;

#if 0
  void get_nodes_from_edge(node_array &array, const edge_index &idx) const;
  void get_nodes_from_face(node_array &array, const face_index &idx) const;
  void get_nodes_from_cell(node_array &array, const cell_index &idx) const;
  void get_edges_from_face(edge_array &array, const face_index &idx) const;
  void get_edges_from_cell(edge_array &array, const cell_index &idx) const;
  void get_faces_from_cell(face_array &array, const cell_index &idx) const;

  void get_neighbor_from_face(cell_index &neighbor, const face_index &idx) const;
#endif

  void locate_node(node_index &node, const Point &p);
  //void locate_edge(edge_index &edge, const Point &p);
  //void locate_face(face_index &face, const Point &p);
  void locate_cell(cell_index &cell, const Point &p);


  void unlocate(Point &result, const Point &p);

  void get_point(Point &result, const node_index &index) const;

  virtual void io(Piostream&);
  static PersistentTypeID type_id;

private:
  int nx_, ny_, nz_;

  static Persistent *maker();
};

} // namespace SCIRun


#endif // SCI_project_MeshRG_h
