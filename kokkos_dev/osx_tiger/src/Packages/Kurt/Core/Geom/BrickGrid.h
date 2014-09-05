#ifndef BRICK_GRID_H
#define BRICK_GRID_H

#include <Core/Containers/Array3.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Vector.h>
#include <Core/Containers/LockingHandle.h>
#include <Packages/Uintah/Core/Grid/ShareAssignArray3.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Semaphore.h>
#include <Packages/Kurt/Core/Geom/BrickGridThread.h>
#include <Packages/Kurt/Core/Geom/GridBrick.h>
#include <vector>

namespace Kurt {
using std::pair;
using std::vector;
using SCIRun::Array3;
using SCIRun::Point;
using SCIRun::BBox;
using SCIRun::Vector;
using SCIRun::LockingHandle;
using SCIRun::Datatype;
using SCIRun::FieldHandle;
using SCIRun::Field;
using SCIRun::LatVolField;
using SCIRun::Piostream;
using SCIRun::Thread;
using Uintah::ShareAssignArray3;
using Uintah::Semaphore;
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

  void OrderBricks(vector<Brick*>& bricks, const Ray& view) const;

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
	    if( iy_ > y_) { iy_--;  return *this; }
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
    void init( LatVolField<Data>& tex );

  unsigned char SETVAL(double val);
  Point min_, max_;
  FieldHandle tex_;
  int brick_size_;
  bool is_fixed_;
  pair<double,double> minmax_;
  Array3<GridBrick*> *bricks_;
  int tex_x_, tex_y_, tex_z_;
  
};




template<class Data> 
void BrickGrid::init( LatVolField<Data>& tex )
{
  int nx,ny,nz;
  int bx,by,bz;

  typename LatVolField<Data>::mesh_type *m = tex.get_typed_mesh().get_rep();
  BBox bb = m->get_bounding_box();
  min_ = bb.min();
  max_ = bb.max();
  
  if( tex.data_at() == Field::CELL ){
    nx = m->get_ni()-1;
    ny = m->get_nj()-1;
    nz = m->get_nk()-1;
  } else {
    nx = m->get_ni();
    ny = m->get_nj();
    nz = m->get_nk();
  }
  tex_x_ = nx; tex_y_ = ny; tex_z_ = nz;

  bx = ceil((nx-1)/(double)(brick_size_ - 1));
  by = ceil((ny-1)/(double)(brick_size_ - 1));
  bz = ceil((nz-1)/(double)(brick_size_ - 1));

  int i,j,k;
  int ix,iy,iz, bs;
  double dx,dy,dz;
  Vector diag(max_ - min_);
  dx = diag.x()/(nx - 1);
  dy = diag.y()/(ny - 1);
  dz = diag.z()/(nz - 1);
  

  if( !bricks_ )
    bricks_ = scinew Array3<GridBrick*>(bx,by,bz);
  else {
    for( i = 0; i < bricks_->dim1(); i++)
      for( j = 0; j < bricks_->dim2(); j++)
	for( k = 0; k < bricks_->dim3(); k++)
	  delete (*bricks_)(i,j,k);

    bricks_->resize(bx,by,bz);
  }
  
  int max_workers = Max(Thread::numProcessors()/2, 4);
  Semaphore* sema = scinew Semaphore( "BrickGrid  semaphore",
					     max_workers); 

  ix = 0;
  bs = brick_size_ - 1;
  for(i = 0; i < bx; i++, ix += bs){
    iy = 0;
    for(j = 0; j < by; j++, iy += bs){
      iz = 0;
      for(k = 0; k < bz; k++, iz+= bs) {
	int padx = 0, pady = 0, padz = 0;
	Array3<unsigned char> *brick_data =
	  scinew Array3<unsigned char>(brick_size_,brick_size_,brick_size_);
	IntVector imin(ix,iy,iz);
	IntVector imax(Min(ix+bs,nx-1),
		       Min(iy+bs,ny-1),
		       Min(iz+bs,nz-1));
	sema->down();
	Thread *thrd = 
	  scinew Thread( scinew LatVolThread<Data>(tex,imin,imax, minmax_,
						  brick_data, sema),
		     "BrickGrid worker");
	thrd->detach();

// 	LatVolThread<Data> no_thread( tex,imin,imax, minmax_, brick_data, sema);
// 	no_thread.run();
	

	if( ix+bs > nx-1) {
	  padx = ix+bs - (nx - 1) ;
	  imax.x(nx - 1);
	}
	if( iy+bs > ny - 1) {
	  pady = iy+bs - (ny - 1) ;
	  imax.y(ny - 1);
	}
	if( iz+bs > nz -1) {
	  padz = iz+bs - (nz - 1) ;
	  imax.z(nz - 1 );
	}
//  	cerr<<"Brick range = ("<<min_ + imin*Vector(dx,dy,dz)<<", "<<
//  	  min_ + imax*Vector(dx,dy,dz)<<")\nBrick min index = ("<<
//  	  imin.x()<<","<<imin.y()<<","<<imin.z()<<")\nBrick max index = ("<<
//  	  imax.x()<<","<<imax.y()<<","<<imax.z()<<")\nWith padding = ("<<
//  	  padx<<","<<pady<<","<<padz<<")\n\n";

	(*bricks_)(i,j,k) =  scinew GridBrick(min_ + imin*Vector(dx,dy,dz),
				       min_ + imax*Vector(dx,dy,dz),
				       imin, imax, padx, pady, padz,
				       0, brick_data);

      }
    }
  }
  sema->down(max_workers);
  if(sema) delete sema;
}



typedef LockingHandle<BrickGrid>  BrickGridHandle;

}// end namespace Kurt
#endif
