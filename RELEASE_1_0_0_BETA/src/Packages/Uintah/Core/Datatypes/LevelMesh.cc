#include "LevelMesh.h"
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Datatypes/FieldAlgo.h>

namespace Uintah {

using SCIRun::IntVector;

PersistentTypeID LevelMesh::type_id("LevelMesh", "MeshBase", maker);

LevelMesh::LevelMesh( GridP  g, int level) : grid_(g), level_(level)
{
  LevelP l = grid_->getLevel( level );
  
  IntVector low, high;
  
  l->getIndexRange( low, high );

  idxLow_ = low;

  nx_ = high.x() - low.x();
  ny_ = high.y() - low.y();
  nz_ = high.z() - low.z();

  min_ = grid_->getLevel( level_ )->getNodePosition( low );
  max_ = grid_->getLevel( level_)->getNodePosition( high );
  
}

LevelMesh::LevelIter::LevelIter(const LevelMesh *m, unsigned i,
			     unsigned j, unsigned k )
  : LevelIndex(i, j, k), mesh_(m)
{
  LevelP l = mesh_->grid_->getLevel( mesh_->level_ );
  patch_ = l->selectPatch( mesh_->idxLow_ + IntVector(i_,j_,k_));
}


BBox LevelMesh::get_bounding_box() const
{
  BBox b;
  b.extend( min_);
  b.extend( max_);
  return b;
}

void 
LevelMesh::get_nodes(node_array &array, cell_index idx) const
{
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
LevelMesh::get_center(Point &result, node_index idx) const
{
  double xgap,ygap,zgap;

  // compute the distance between slices
  xgap = (max_.x()-min_.x())/(nx_-1);
  ygap = (max_.y()-min_.y())/(ny_-1);
  zgap = (max_.z()-min_.z())/(nz_-1);
  
  // return the node_index converted to object space
  result.x(min_.x()+idx.i_*xgap);
  result.y(min_.y()+idx.j_*ygap);
  result.z(min_.z()+idx.k_*zgap);
}

void
LevelMesh::get_point(Point &result, node_index index) const
{ 
  get_center(result,index);
}

void 
LevelMesh::get_center(Point &result, cell_index idx) const
{
  node_array nodes;
  Point min,max;

  // get the node_indeces inside of this cell
  get_nodes(nodes,idx);

  // convert the min and max nodes of the cell into object space points
  get_point(min,nodes[0]);
  get_point(max,nodes[7]);

  // return the point half way between min and max
  result.x(min.x()+(max.x()-min.x())*.5);
  result.y(min.y()+(max.y()-min.y())*.5);
  result.z(min.z()+(max.z()-min.z())*.5);
}

bool
LevelMesh::locate(cell_index &cell, const Point &p) const
{
  IntVector idx = grid_->getLevel(level_)->getCellIndex( p );

  cell.i_ = (unsigned)idx.x() - idxLow_.x();
  cell.j_ = (unsigned)idx.y() - idxLow_.y();
  cell.k_ = (unsigned)idx.z() - idxLow_.z();

  return true;
}

bool
LevelMesh::locate(node_index &node, const Point &p) const
{
  node_array nodes;     // storage for node_indeces
  cell_index cell;
  double max;
  int loop;

  // locate the cell enclosing the point (including weights)
  if (!locate(cell,p)) return false;
  weight_array w;
  calc_weights(this, cell, p, w);

  // get the node_indeces in this cell
  get_nodes(nodes,cell);

  // find, and return, the "heaviest" node
  max = w[0];
  loop=1;
  while (loop<8) {
    if (w[loop]>max) {
      max=w[loop];
      node=nodes[loop];
    }
  }
  return true;
}

#define LEVELMESH_VERSION 1

void
LevelMesh::io(Piostream& stream)
{
  stream.begin_class(type_name().c_str(), LEVELMESH_VERSION);

  // IO data members, in order
//   Pio(stream, grid_);
//   Pio(stream, level_);
//   Pio(stream, nx_);
//   Pio(stream, ny_);
//   Pio(stream, nz_);
//   Pio(stream, min_);
//   Pio(stream, max_);

  // We need to figure out how to deal with grid_
  NOT_FINISHED("LevelMesh::io");

  stream.end_class();
}

const string 
LevelMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "LevelMesh";
  return name;
}


}  // end namespace Uintah
