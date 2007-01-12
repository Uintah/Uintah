
#include "BrickGrid.h"
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Thread.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <Core/Util/NotFinished.h>
#include <Packages/Uintah/Core/Grid/ShareAssignArray3.h>
#include <iostream>
#include <string>
#include <vector>
#include <functional>


namespace Kurt {
using std::cerr;
using std::endl;
using std::vector;
using std::binary_function;
using std::sort;
using SCIRun::Thread;
using SCIRun::Semaphore;
using SCIRun::Point;
using SCIRun::IntVector;
using SCIRun::Vector;
using SCIRun::Min;
using SCIRun::LatVolField;
using SCIRun::field_minmax;
using SCIRun::Field;
using Uintah::ShareAssignArray3;
using SCIRun::Min;
using SCIRun::Max;
using SCIRun::Array3;

BrickGrid::BrickGrid(FieldHandle tex, int bricksize, bool fixed,
		     double min, double max) :
  tex_(tex), brick_size_(bricksize), is_fixed_( fixed ),
  minmax_(min, max), bricks_(0)
{
}

void BrickGrid::init() 
{
  const std::string field_type = tex_->get_type_name(0);
  if( field_type != "LatVolField"){
    cerr<<"BrickGrid not compatible with field type "<<field_type<<endl;
    return;
  }
  const std::string data_type = tex_->get_type_name(1);
  
  if( field_type == "LatVolField" ) {
    if( data_type == "double" ) {
      LatVolField<double> *fld =
	dynamic_cast<LatVolField<double>*>(tex_.get_rep());
      if( !is_fixed_ )
	field_minmax(*fld, minmax_);
      init(*fld);
    } else if (data_type == "int" ) {
      LatVolField<int> *fld =
	dynamic_cast<LatVolField<int>*>(tex_.get_rep());
      if( !is_fixed_ )
	field_minmax(*fld, minmax_);
      init(*fld);
    } else if (data_type == "short" ) {
      LatVolField<short> *fld =
	dynamic_cast<LatVolField<short>*>(tex_.get_rep());
      if( !is_fixed_ )
	field_minmax(*fld, minmax_);
      init(*fld);
    } else if (data_type == "unsigned_char" ) {
      LatVolField<unsigned char> *fld =
	dynamic_cast<LatVolField<unsigned char>*>(tex_.get_rep());
      if( !is_fixed_ )
	field_minmax(*fld, minmax_);
      init(*fld);
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


class gr_distance 
{
public:
  gr_distance(const Ray& view ): view_(view){}
  bool operator()(Brick *b1, Brick *b2){ 
    double d1 = (view_.origin() - b1->get_center()).length(); 
    double d2 = (view_.origin() - b2->get_center()).length(); 
    return (d1 > d2);
  }
protected:
  Ray view_;
};


void
BrickGrid::OrderBricks(vector<Brick*>& bricks, const Ray& view) const
{
  for(int i=0; i < nx(); i++)
    for(int j = 0; j < ny(); j++)
      for(int k = 0; k < nz(); k++)
	bricks.push_back( (*bricks_)(i,j,k) );
  
  sort(bricks.begin(), bricks.end(), gr_distance(view));
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




void BrickGrid::io(Piostream&)
{
    // Nothing for now...
  NOT_FINISHED("VolumeRenderer::io");
}
	  
    
} // namespace Kurt
