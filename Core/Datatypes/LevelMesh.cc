#include "LevelMesh.h"
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Core/Util/NotFinished.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/BBox.h>
#include <Core/Datatypes/FieldAlgo.h>

namespace Uintah {

using SCIRun::IntVector;

PersistentTypeID LevelMesh::type_id("LevelMesh", "Mesh", maker);
LevelMesh:: LevelMesh( LevelMesh* mh, int mx, int my, int mz,
	     int x, int y, int z) :
  grid_(mh->grid_), level_(mh->level_), 
  idxLow_(mh->idxLow_ + IntVector( mx, my, mz)),
  nx_(x), ny_(y), nz_(z), 
  min_(mh->grid_->getLevel( mh->level_ )->getNodePosition( idxLow_)),
  max_(mh->grid_->getLevel( mh->level_ )->
       getNodePosition( idxLow_ + IntVector(x,y,z)))
{
  cerr<<"in LevelMesh constructor \n";
  cerr<<"level = "<<level_<<",\nidxLow = "<<idxLow_<<","
      <<"nx,ny,nz = "<<nx_<<","<<ny_<<","<<nz_<<endl;
  cerr<<"min = "<< min_<<", max = "<< max_<<endl;
}

void
LevelMesh::init()
{
  LevelP l = grid_->getLevel( level_ );
  BBox bb;
  IntVector low, high;
  
  l->findIndexRange( low, high );
  l->getSpatialRange( bb );
  idxLow_ = low;

  nx_ = high.x() - low.x();
  ny_ = high.y() - low.y();
  nz_ = high.z() - low.z();

  min_ = bb.min();
  max_ = bb.max();

  cerr<<"in LevelMesh constructor \n";
  cerr<<"level = "<<level_<<",\nidxLow = "<<idxLow_<<","
      <<"nx,ny,nz = "<<nx_<<","<<ny_<<","<<nz_<<endl;
  cerr<<"min = "<< min_<<", max = "<< max_<<endl;
}  

LevelMesh::LevelIndex::LevelIndex(const LevelMesh *m, int i,
				int j, int k ) :
  mesh_(m), i_(m->idxLow_.x() + i), j_(m->idxLow_.y() + j),
  k_(m->idxLow_.z() + k)
{
  //  cerr<<"index i_,j_,k_ = ("<<i_<<","<<j_<<","<<k_<<")\n";

  patch_ = mesh_->grid_->getLevel( mesh_->level_ )->
    selectPatchForNodeIndex( IntVector(i_,j_,k_));
}
LevelMesh::CellIndex::CellIndex(const LevelMesh *m, int i,
				int j, int k ) :
  LevelMesh::LevelIndex(m, i, j, k)
{
  patch_ = mesh_->grid_->getLevel(mesh_->level_)-> 
    selectPatchForCellIndex(IntVector(i_,j_,k_));
}  

BBox LevelMesh::get_bounding_box() const
{
  BBox b;
  b.extend( min_);
  b.extend( max_);
  return b;
}


void
LevelMesh::transform(Transform &)
{
  ASSERTFAIL("Not Transformable mesh");
}


void 
LevelMesh::get_nodes(Node::array_type &array, Cell::index_type idx) const
{
  array.resize(8);
  array[0].i_ = idx.i_; array[0].j_ = idx.j_;  array[0].k_ = idx.k_; 
  array[0].mesh_ = idx.mesh_;  array[0].patch_ = idx.patch_;
  array[1].i_ = idx.i_+1; array[1].j_ = idx.j_; array[1].k_ = idx.k_; 
  array[1].mesh_ = idx.mesh_; array[1].patch_ = idx.patch_;
  array[2].i_ = idx.i_+1; array[2].j_ = idx.j_+1; array[2].k_ = idx.k_; 
  array[2].mesh_ = idx.mesh_; array[2].patch_ = idx.patch_;
  array[3].i_ = idx.i_; array[3].j_ = idx.j_+1; array[3].k_ = idx.k_; 
  array[3].mesh_ = idx.mesh_; array[3].patch_ = idx.patch_;
  array[4].i_ = idx.i_; array[4].j_ = idx.j_; array[4].k_ = idx.k_+1;
  array[4].mesh_ = idx.mesh_; array[4].patch_ = idx.patch_;
  array[5].i_ = idx.i_+1; array[5].j_ = idx.j_; array[5].k_ = idx.k_+1;
  array[5].mesh_ = idx.mesh_; array[5].patch_ = idx.patch_;
  array[6].i_ = idx.i_+1; array[6].j_ = idx.j_+1; array[6].k_ = idx.k_+1;
  array[6].mesh_ = idx.mesh_; array[6].patch_ = idx.patch_;
  array[7].i_ = idx.i_; array[7].j_ = idx.j_+1; array[7].k_ = idx.k_+1;
  array[7].mesh_ = idx.mesh_; array[7].patch_ = idx.patch_;
  for(int i = 1; i < 8; i++){
    IntVector index(array[i].i_, array[i].j_, array[i].k_);
    if( !(array[i].patch_->containsNode( index ) ) )
      array[i].patch_ =
	array[i].mesh_->
	  grid_->getLevel(array[i].mesh_->level_)->
	               selectPatchForNodeIndex( index ); 
  }
}

void 
LevelMesh::get_center(Point &result, Node::index_type idx) const
{
  double xgap,ygap,zgap;

  // compute the distance between slices
  xgap = (max_.x()-min_.x())/(nx_-1);
  ygap = (max_.y()-min_.y())/(ny_-1);
  zgap = (max_.z()-min_.z())/(nz_-1);
  
  // return the Node::index_type converted to object space
  result.x(min_.x()+idx.i_*xgap);
  result.y(min_.y()+idx.j_*ygap);
  result.z(min_.z()+idx.k_*zgap);
}

void
LevelMesh::get_point(Point &result, Node::index_type index) const
{ 
  get_center(result,index);
}

void 
LevelMesh::get_center(Point &result, Cell::index_type idx) const
{
  Node::array_type nodes;
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
LevelMesh::locate(Cell::index_type &cell, const Point &p) const
{
  IntVector l, h;
  grid_->getLevel(level_)->findCellIndexRange(l, h);
  IntVector idx = grid_->getLevel(level_)->getCellIndex( p );
  if( idx.x() >= l.x() && idx.y() >= l.y() && idx.z() >= l.z()
      && idx.x() < h.x() && idx.y() < h.y() && idx.z() < h.z() ){
    cell.i_ = (int)idx.x();
    cell.j_ = (int)idx.y();
    cell.k_ = (int)idx.z();
    cell.mesh_ = this;
    cell.patch_ = grid_->getLevel(level_)->selectPatchForCellIndex( idx);
    return true;
  } else {
    return false;
  }
}

bool
LevelMesh::locate(Node::index_type &node, const Point &p) const
{ 
  Node::array_type nodes;     // storage for node_indeces
  Cell::index_type cell;
  double max;
  int loop;
  
  if( grid_->getLevel(level_)->containsPoint(p) ) {
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
    node.mesh_ = this;
    node.patch_ = grid_->getLevel(level_)->
      selectPatchForNodeIndex(IntVector(node.i_, node.j_, node.k_)  );
    return true;
  } else {
    return false;
  }
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

void
LevelMesh::get_weights(const Point &p,
			Cell::array_type &l, vector<double> &w)
{
  Cell::index_type idx;
  if (locate(idx, p))
  {
    l.push_back(idx);
    w.push_back(1.0);
  }
}

void
LevelMesh::get_weights(const Point &p,
			Node::array_type &l, vector<double> &w)
{
  Cell::index_type idx;
  if (locate(idx, p))
  {
    get_nodes( l, idx );
    w.resize(l.size());
    vector<double>::iterator wit = w.begin();
    Node::array_type::iterator it = l.begin();

    Point np, pmin, pmax;
    get_point(pmin, l[0]);
    get_point(pmax, l[6]);

    Vector diag(pmax - pmin);

    while( it != l.end()) {
      Node::index_type ni = *it;
      ++it;
      get_point(np, ni);
      *wit = ( 1 - fabs(p.x() - np.x())/diag.x() ) *
	( 1 - fabs(p.y() - np.y())/diag.y() ) *
	( 1 - fabs(p.z() - np.z())/diag.z() );
      ++wit;
    }
  }
}



const SCIRun::TypeDescription*
LevelMesh::get_type_description() const
{
  return SCIRun::get_type_description((LevelMesh *)0);
}

}

namespace SCIRun {


const TypeDescription*
get_type_description(Uintah::LevelMesh *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("LevelMesh",
				TypeDescription::cc_to_h(__FILE__),
					"Uintah");
  }
  return td;
}

const TypeDescription*
get_type_description(Uintah::LevelMesh::Node *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("LevelMesh::Node",
				TypeDescription::cc_to_h(__FILE__),
				"Uintah");
  }
  return td;
}

const TypeDescription*
get_type_description(Uintah::LevelMesh::Edge *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("LevelMesh::Edge",
				TypeDescription::cc_to_h(__FILE__),
				"Uintah");
  }
  return td;
}

const TypeDescription*
get_type_description(Uintah::LevelMesh::Face *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("LevelMesh::Face",
				TypeDescription::cc_to_h(__FILE__),
				"Uintah");
  }
  return td;
}

const TypeDescription*
get_type_description(Uintah::LevelMesh::Cell *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("LevelMesh::Cell",
				TypeDescription::cc_to_h(__FILE__),
				"Uintah");
  }
  return td;
}


}  // end namespace Uintah
