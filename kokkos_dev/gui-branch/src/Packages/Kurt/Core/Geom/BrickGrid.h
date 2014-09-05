#ifndef BRICK_GRID_H
#define BRICK_GRID_H

#include <Packages/Kurt/Core/Geom/GridBrick.h>
#include <Core/Containers/Array3.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Packages/Uintah/Core/Datatypes/LevelField.h>
#include <Packages/Uintah/Core/Grid/ShareAssignArray3.h>
namespace Kurt {

using SCIRun::Array3;
using SCIRun::Point;
using SCIRun::BBox;
using SCIRun::LockingHandle;
using SCIRun::Datatype;
using SCIRun::FieldHandle;
using SCIRun::LatticeVol;
using SCIRun::Piostream;
using Uintah::ShareAssignArray3;
using Uintah::LevelField;
class GridVolRen;


class BrickGrid : public Datatype {
  friend class GridVolRen;
public:
  BrickGrid(FieldHandle tex, int bricksize, bool fixed,
	    double min, double max);
  virtual ~BrickGrid() { 
    int i,j,k;
    if( bricks_ ){
      for( i = 0; i < bricks_->dim1(); i++)
	for( j = 0; j < bricks_->dim2(); j++)
	  for( k = 0; k < bricks_->dim3(); k++)
	    delete (*bricks_)(i,j,k);
    
      delete bricks_;
    }
  }

  virtual void io(Piostream &);

  GridBrick*& operator()(int i, int j, int k) const
    { return (*bricks_)(i,j,k); }
  Point& get_min_point(){ return min_; }
  Point& get_max_point(){ return max_; }
  void get_bounds( BBox& bb ) { bb.extend(min_); bb.extend(max_); }
  void get_range( pair<double,double>& range){ range = minmax_; }
  void init();
  
  int nx() const { return bricks_->dim1(); }
  int ny() const { return bricks_->dim2(); }
  int nz() const { return bricks_->dim3(); }

  class iterator {
  public:
    iterator(const BrickGrid* bg, int x, int y, int z, const Ray& r,
	     int ix = 0, int iy = 0, int iz = 0)
      : x_(x), y_(y), z_(z), ix_(ix), iy_(iy), iz_(iz),
      bg_(bg), view_(r) 
      {
	if(ix_ != -1 ){
	  if(x_ == 0) ix_ = bg_->nx()-1;
	  else { ix_ = 0; }
	  if(y_ == 0) iy_ = bg_->ny()-1;
	  else { iy_ = 0; }
	  if(z_ == 0) iz_ = bg_->nz()-1;
	  else { iz_ = 0; }
	}
      }
    iterator(const iterator& copy) 
       : x_(copy.x_), y_(copy.y_), z_(copy.z_), ix_(copy.ix_),
      iy_(copy.iy_), iz_(copy.iz_), bg_(copy.bg_), view_(copy.view_) {}
     
    ~iterator(){}
    iterator& operator=(const iterator& it2)
    { 
      ix_ = it2.ix_; iy_ = it2.iy_; iz_ = it2.iz_;
      x_ = it2.x_; y_ = it2.y_; z_ = it2.z_;
      bg_ = it2.bg_; view_ = it2.view_; return *this; 
    }
    
    bool operator==(const iterator& it2) const
    { return (x_ == it2.x_ && y_ == it2.y_ && z_ == it2.z_ &&
	      ix_ == it2.ix_ && iy_ == it2.iy_ && iz_ == it2.iz_ &&
	      bg_ == it2.bg_ && view_.origin() == it2.view_.origin() &&
	      view_.direction() == it2.view_.direction()); }

    bool operator!=(const iterator& it2) const
    { return (x_ != it2.x_ || y_ != it2.y_ || z_ != it2.z_ ||
	      ix_ != it2.ix_ || iy_ != it2.iy_ || iz_ != it2.iz_ ||
	      bg_ != it2.bg_ || view_.origin() != it2.view_.origin() ||
	      view_.direction() != it2.view_.direction()); }

    GridBrick*& operator*() { return (*bg_)(ix_, iy_, iz_); }

    iterator operator++()
      {
	static bool rz = false, ry = false, rx = false;
	if( iz_ < z_) iz_++;
	if( iz_ > z_) { iz_--; return *this;}
	if( iz_ == z_ ) {
	  if( !rz ) {
	    iz_ = bg_->nz() - 1;
	    rz = true;
	  } else {
	    rz = false;
	    iz_ = ((z_ == 0) ?  bg_->nz() -1 : 0);
	    if( iy_ < y_) iy_++;
	    if( iy_ > y_) { iy_--; return *this; }
	    if( iy_ == y_ ) {
	      if( !ry ) {
		iy_ = bg_->ny() - 1;
		ry = true;
	      } else {
		iy_ = ((y_ == 0) ? bg_->ny() - 1 : 0);
		ry = false;
		if( ix_ < x_) ix_++;
		if( ix_ > x_) { ix_--; return *this; }
		if( ix_ == x_ ) {
		  if( !rx ) {
		    ix_ = bg_->nx() - 1;
		    rx = true;
		  } else {
		    rx = false;
		    ix_ = -1;
		    iy_ = -1;
		    iz_ = -1;
		  }
		}
	      }
	    }
	  }
	}
	return *this;
      }
    
    iterator operator++(int)
      {
	BrickGrid::iterator old(*this);
	++(*this);
	return old;
      }

    IntVector getIndex() const { return IntVector(ix_,iy_,iz_); }
  private:
    int x_, y_, z_;
    int ix_, iy_, iz_;
    const BrickGrid* bg_;
    Ray view_;
  };

  iterator begin( Ray& view ) const;
  iterator end( Ray& view) const ;
    
private:
  template <class Data>
    void lat_vol_init( LatticeVol<Data>& tex );

  template <class Data>
    void level_field_init( LevelField<Data>& tex );

  unsigned char SETVAL(double val);
  bool is_fixed_;
  Point min_, max_;
  pair<double,double> minmax_;
  Array3<GridBrick*> *bricks_;
  FieldHandle tex_;
  int brick_size_;
  
};

typedef LockingHandle<BrickGrid>  BrickGridHandle;

}// end namespace Kurt
#endif
