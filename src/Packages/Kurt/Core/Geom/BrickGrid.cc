
#include "BrickGrid.h"
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Thread.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <Packages/Uintah/Core/Datatypes/LevelField.h>
#include <Packages/Uintah/Core/Grid/ShareAssignArray3.h>
#include <Packages/Kurt/Core/Geom/BrickGridThread.h>
#include <iostream>

using std::cerr;
using std::endl;

namespace Kurt {
using SCIRun::Thread;
using SCIRun::Semaphore;
using SCIRun::Point;
using SCIRun::IntVector;
using SCIRun::Vector;
using SCIRun::Min;
using SCIRun::LatticeVol;
using SCIRun::field_minmax;
using SCIRun::Field;
using Uintah::LevelField;
using Uintah::ShareAssignArray3;
using SCIRun::Min;
using SCIRun::Max;

BrickGrid::BrickGrid(FieldHandle tex, int bricksize) :
  tex_(tex), brick_size_(bricksize)
{
}

void BrickGrid::init() 
{
  const string field_type = tex_->get_type_name(0);
  if( field_type != "LatticeVol" && field_type != "LevelField"){
    cerr<<"BrickGrid not compatible with field type "<<field_type<<endl;
    return;
  }
  const string data_type = tex_->get_type_name(1);
  
  if( field_type == "LatticeVol" ) {
    if( data_type == "double" ) {
      LatticeVol<double> *fld =
	dynamic_cast<LatticeVol<double>*>(tex_.get_rep());
      field_minmax(*fld, minmax_);
      lat_vol_init(*fld);
    } else if (data_type == "int" ) {
      LatticeVol<int> *fld =
	dynamic_cast<LatticeVol<int>*>(tex_.get_rep());
      field_minmax(*fld, minmax_);
      lat_vol_init(*fld);
    } else if (data_type == "short" ) {
      LatticeVol<short> *fld =
	dynamic_cast<LatticeVol<short>*>(tex_.get_rep());
      field_minmax(*fld, minmax_);
      lat_vol_init(*fld);
    } else if (data_type == "unsigned_char" ) {
      LatticeVol<unsigned char> *fld =
	dynamic_cast<LatticeVol<unsigned char>*>(tex_.get_rep());
      field_minmax(*fld, minmax_);
      lat_vol_init(*fld);
    }
  } else {
    if( data_type == "double" ) {
      LevelField<double> *fld =
	dynamic_cast<LevelField<double>*>(tex_.get_rep());
      fld->minmax( minmax_ );
      level_field_init(*fld);
    } else if( data_type == "float" ) {
      LevelField<float> *fld =
	dynamic_cast<LevelField<float>*>(tex_.get_rep());
      fld->minmax(minmax_);
      level_field_init(*fld);
    } else if( data_type == "long" ) {
      LevelField<long> *fld =
	dynamic_cast<LevelField<long>*>(tex_.get_rep());
      fld->minmax(minmax_);
      level_field_init(*fld);
    }
  }
}

unsigned char
BrickGrid::SETVAL(double val)
{
  double v = (val - minmax_.first)*255/(minmax_.second - minmax_.first);
  if ( v < 0 ) return 0;
  else if (v > 255) return (unsigned char)255;
  else return (unsigned char)v;
}

template<class Data> 
void BrickGrid::level_field_init(LevelField<Data>& tex)
{
  int nx,ny,nz;
  int bx,by,bz;

  const string data_type = tex.get_type_name(1);
  typename LevelField<Data>::mesh_type *m = tex.get_typed_mesh().get_rep();

  min_ = m->get_min();
  max_ = m->get_max();
  
  if( tex.data_at() == Field::CELL ){
    nx = m->get_nx()-1;
    ny = m->get_ny()-1;
    nz = m->get_nz()-1;
  } else {
    nx = m->get_nx();
    ny = m->get_ny();
    nz = m->get_nz();
  }

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
  
 
  bricks_ = new Array3<GridBrick*>(bx,by,bz);

  ix = 0;
  bs = brick_size_ - 1;
  for(i = 0; i < bx; i++, ix += bs){
    iy = 0;
    for(j = 0; j < by; j++, iy += bs){
      iz = 0;
      for(k = 0; k < bz; k++, iz+= bs) {
	int padx = 0, pady = 0, padz = 0;
	Array3<unsigned char> *brick_data =
	  new Array3<unsigned char>(brick_size_,brick_size_,brick_size_);
	IntVector imin(ix,iy,iz);
	IntVector imax(Min(ix+bs,nx-1),
		       Min(iy+bs,ny-1),
		       Min(iz+bs,nz-1));
	if( ix+bs > nx-1) {
	  padx = ix+bs - nx - 1 ;
	  imax.x(nx - 1);
	}
	if( iy+bs > ny - 1) {
	  pady = iy+bs - ny - 1 ;
	  imax.y(ny - 1);
	}
	if( iz+bs > nz -1) {
	  padz = iz+bs - nz - 1 ;
	  imax.z(nz - 1 );
	}

	(*bricks_)(i,j,k) =
	  new GridBrick(min_ + imin*Vector(dx,dy,dz),
			min_ + imax*Vector(dx,dy,dz),
			imin, imax, padx, pady, padz, 0, brick_data);
      }
    }
  }

  int max_workers = Max(Thread::numProcessors()/2, 4);
  Semaphore* sema = scinew Semaphore( "BrickGrid  semaphore",
					     max_workers); 

  vector<ShareAssignArray3<Data> > tdata = tex.fdata();
  vector<ShareAssignArray3<Data> >::iterator vit = tdata.begin();
  vector<ShareAssignArray3<Data> >::iterator vit_end = tdata.end();
  IntVector offset( (*vit).getLowIndex() );
  for(;vit != vit_end; ++ vit){
    sema->down();
    Thread *thrd = 
      scinew Thread( scinew BrickGridThread<Data>(*vit, minmax_,
						  bricks_, offset, sema),
		     "BrickGrid worker");
    thrd->detach();
  }
  sema->down(max_workers);
  if(sema) delete sema;
  
}

template<class Data> 
void BrickGrid::lat_vol_init( LatticeVol<Data>& tex )
{
  int nx,ny,nz;
  int bx,by,bz;

  typename LatticeVol<Data>::mesh_type *m = tex.get_typed_mesh().get_rep();
  min_ = m->get_min();
  max_ = m->get_max();
  
  if( tex.data_at() == Field::CELL ){
    nx = m->get_nx()-1;
    ny = m->get_ny()-1;
    nz = m->get_nz()-1;
  } else {
    nx = m->get_nx();
    ny = m->get_ny();
    nz = m->get_nz();
  }

  cerr<<"Texture Data size = ("<<nx<<","<<ny<<","<<nz<<
    ")\nTexture min point is "<< min_<<
    "\nTexture max point is "<<max_<<endl;

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
  
 
  bricks_ = new Array3<GridBrick*>(bx,by,bz);
  
  ix = 0;
  bs = brick_size_ - 1;
  for(i = 0; i < bx; i++, ix += bs){
    iy = 0;
    for(j = 0; j < by; j++, iy += bs){
      iz = 0;
      for(k = 0; k < bz; k++, iz+= bs) {
	int padx = 0, pady = 0, padz = 0;
	Array3<unsigned char> *brick_data =
	  new Array3<unsigned char>(brick_size_,brick_size_,brick_size_);
	IntVector imin(ix,iy,iz);
	IntVector imax(Min(ix+bs,nx-1),
		       Min(iy+bs,ny-1),
		       Min(iz+bs,nz-1));
	fill_brick_from_lat_vol_data(tex, imin, imax, brick_data);
	if( ix+bs > nx-1) {
	  padx = ix+bs - nx - 1 ;
	  imax.x(nx - 1);
	}
	if( iy+bs > ny - 1) {
	  pady = iy+bs - ny - 1 ;
	  imax.y(ny - 1);
	}
	if( iz+bs > nz -1) {
	  padz = iz+bs - nz - 1 ;
	  imax.z(nz - 1 );
	}
	cerr<<"Brick range = ("<<min_ + imin*Vector(dx,dy,dz)<<", "<<
	  min_ + imax*Vector(dx,dy,dz)<<")\nBrick min index = ("<<
	  imin.x()<<","<<imin.y()<<","<<imin.z()<<")\nBrick max index = ("<<
	  imax.x()<<","<<imax.y()<<","<<imax.z()<<")\nWith padding = ("<<
	  padx<<","<<pady<<","<<padz<<")\n\n";

	(*bricks_)(i,j,k) =
	  new GridBrick(min_ + imin*Vector(dx,dy,dz),
			min_ + imax*Vector(dx,dy,dz),
			imin, imax, padx, pady, padz, 0, brick_data);
      }
    }
  }
}
template <class Data>
void BrickGrid::fill_brick_from_lat_vol_data(LatticeVol<Data>& tex,
					     IntVector imin,
					     IntVector imax,
					     Array3<unsigned char> *brick_data)
{
  using std::cerr;
  using std::endl;
  int ii,jj,kk;
  
  int xsize_ = imax.x() - imin.x()+1;
  int ysize_ = imax.y() - imin.y()+1;
  int zsize_ = imax.z() - imin.z()+1;

  typename LatticeVol<Data>::mesh_type *m = tex.get_typed_mesh().get_rep();
  if( tex_->data_at() == Field::CELL){
    typename LatticeVol<Data>::mesh_type mesh(m, imin.x(), imin.y(), imin.z(), 
			       xsize_+1, ysize_+1, zsize_+1);
    typename LatticeVol<Data>::mesh_type::Cell::iterator it = mesh.cell_begin();
    for(ii = 0; ii < xsize_; ii++){
      for(jj = 0; jj < ysize_; jj++) {
	for(kk = 0; kk < zsize_; kk++) {
	  (*brick_data)(kk,jj,ii) = SETVAL( tex.fdata()[*it] );
	  ++it;
	}
      }
    }
  } else {
//     typename LatticeVol<Data>::mesh_type mesh(m, imin.x(), 
// 					      imin.y(), imin.z(), 
// 					      xsize_, ysize_, zsize_);
//     typename LatticeVol<Data>::mesh_type::Node::iterator it = 
//       mesh.node_begin();
//     for(ii = 0; ii < xsize_; ii++){
//       for(jj = 0; jj < ysize_; jj++) {
// 	for(kk = 0; kk < zsize_; kk++) {
// 	  (*brick_data)(kk,jj,ii) = SETVAL( tex.fdata()[*it] );
// 	  ++it;
// 	}
//       }
    int i,j,k;
    int max_i = imax.x() + 1;
    int max_j = imax.y() + 1;
    int max_k = imax.z() + 1;
    
    ii = jj = kk = 0;
    
    for( i = imin.x(); i < max_i; i++, ii++){
      jj = 0;
      for(j = imin.y(); j < max_j; j++, jj++) {
	kk = 0;
	for(k = imin.z(); k < max_k; k++, kk++) {
	  typename LatticeVol<Data>::mesh_type::NodeIndex idx(i,j,k);
	  (*brick_data)(kk,jj,ii) =  SETVAL( tex.fdata()[idx] );
	}
      }
    }
  }
}

				

BrickGrid::iterator BrickGrid::begin( Ray& view ) const
{
  double dx,dy,dz;
  Vector diag(max_ - min_);
  dx = diag.x()/nx();
  dy = diag.y()/ny();
  dz = diag.z()/nz();

  int ix = (view.origin().x() - min_.x())/dx;
  int iy = (view.origin().y() - min_.y())/dy;
  int iz = (view.origin().z() - min_.z())/dz;

  ix = ((ix < 0) ? 0:((ix > nx()-1) ? nx()-1: ix));
  iy = ((iy < 0) ? 0:((iy > ny()-1) ? ny()-1: iy));
  iz = ((iz < 0) ? 0:((iz > nz()-1) ? nz()-1: iz));

  return BrickGrid::iterator(this, ix, iy, iz, view);
					 
}
  
BrickGrid::iterator BrickGrid::end( Ray& view ) const
{
  double dx,dy,dz;
  Vector diag(max_ - min_);
  dx = diag.x()/nx();
  dy = diag.y()/ny();
  dz = diag.z()/nz();

  int ix = (view.origin().x() - min_.x())/dx;
  int iy = (view.origin().y() - min_.y())/dy;
  int iz = (view.origin().z() - min_.z())/dz;

  ix = ((ix < 0) ? 0:((ix > nx()-1) ? nx()-1: ix));
  iy = ((iy < 0) ? 0:((iy > ny()-1) ? ny()-1: iy));
  iz = ((iz < 0) ? 0:((iz > nz()-1) ? nz()-1: iz));

  return BrickGrid::iterator(this, ix,iy,iz, view, -1,-1,-1);
					 
}

BrickGrid::iterator BrickGrid::iterator::operator++()
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

BrickGrid::iterator BrickGrid::iterator::operator++(int)
{
  BrickGrid::iterator old(*this);
  ++(*this);
  return old;
}


void BrickGrid::io(Piostream&)
{
    // Nothing for now...
  NOT_FINISHED("VolumeRenderer::io");
}
	  
    
} // namespace Kurt
