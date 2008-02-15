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
 *  ScalarFieldRG.h: Templated Scalar Fields defined on a Regular grid
 *
 *  Written by:
 *   Michael Callahan (Steve Parker, Dave Weinstein)
 *   Department of Computer Science
 *   University of Utah
 *   January 2000
 *
 *  Copyright (C) 2000 SCI Group
 *
 */

#ifndef SCI_project_ScalarFieldRGT_h
#define SCI_project_ScalarFieldRGT_h 1

#include <Core/Math/MiscMath.h>
#include <FieldConverters/Core/Datatypes/ScalarFieldRGBase.h>
#include <Core/Containers/Array3.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>

namespace FieldConverters {

using namespace SCIRun;

template <class T>
class SCICORESHARE ScalarFieldRGT : public ScalarFieldRGBase {
public:
  Array3<T> grid;

  ScalarFieldRGT(int x, int y, int z);
  ScalarFieldRGT(const ScalarFieldRGT &);
  virtual ~ScalarFieldRGT();
  virtual ScalarField* clone();

  virtual void compute_minmax();
  virtual Vector gradient(const Point&);
  virtual int interpolate(const Point&, double&, double epsilon1=1.e-6, double epsilon2=1.e-6);
  virtual int interpolate(const Point&, double&, int&, double epsilon1=1.e-6, double epsilon2=1.e-6, int exhaustive=0);

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
    
  virtual double get_value( const Point& ivoxel );
  double get_value( int x, int y, int z );

  Vector get_normal( const Point& ivoxel );
  Vector get_normal( int x, int y, int z );

  Vector gradient(int x, int y, int z);

  // this has to be called before augmented stuff (base class)
    
  virtual void fill_gradmags();

private:

  static Persistent *maker();
};


template <> ScalarFieldRGT<double>::ScalarFieldRGT(int x, int y, int z);
template <> ScalarFieldRGT<int>::ScalarFieldRGT(int x, int y, int z);
template <> ScalarFieldRGT<short>::ScalarFieldRGT(int x, int y, int z);
template <> ScalarFieldRGT<char>::ScalarFieldRGT(int x, int y, int z);

template <> PersistentTypeID ScalarFieldRGT<double>::type_id;
template <> PersistentTypeID ScalarFieldRGT<int>::type_id;
template <> PersistentTypeID ScalarFieldRGT<short>::type_id;
template <> PersistentTypeID ScalarFieldRGT<char>::type_id;


template <class T>
Persistent *
ScalarFieldRGT<T>::maker()
{
  return scinew ScalarFieldRGT<T>(0, 0, 0);
}

template <class T>
ScalarFieldRGT<T>::ScalarFieldRGT(const ScalarFieldRGT<T>& copy)
  : ScalarFieldRGBase(copy)
{
  grid.copy(copy.grid);
}

template <class T>
ScalarFieldRGT<T>::~ScalarFieldRGT()
{
}

template <class T>
double
ScalarFieldRGT<T>::get_value( const Point& ivoxel )
{
  return (double) grid( ivoxel.x(), ivoxel.y(), ivoxel.z() );
}

template <class T>
double
ScalarFieldRGT<T>::get_value( int x, int y, int z )
{
  return (double) grid( x, y, z );
}

#define ScalarFieldRGT_VERSION 3

template <class T>
void ScalarFieldRGT<T>::io(Piostream& stream)
{
  int version=stream.begin_class(type_id.type.c_str(), ScalarFieldRGT_VERSION);
  if(version == 1) {
    // From before, when the ScalarFieldRGBase didn't exist...
    ScalarField::io(stream);

    // Save these since the ScalarField doesn't
    SCIRun::Pio(stream, bmin);
    SCIRun::Pio(stream, bmax);
    if(stream.reading()){
      have_bounds=1;
      diagonal=bmax-bmin;
    }

    // Save the rest..
    SCIRun::Pio(stream, nx);
    SCIRun::Pio(stream, ny);
    SCIRun::Pio(stream, nz);
  } else {
    // Do the base class first...
    ScalarFieldRGBase::io(stream);
  }

  if ( stream.reading() ) {
    std::cerr << "reading...version " << version << std::endl;
    if  ( version > 2 )
      SCIRun::Pio(stream, separate_raw);
    else
      separate_raw = 0;

    if ( separate_raw == 1) {
      SCIRun::Pio(stream,raw_filename);
      string filename;
      if ( raw_filename[0] == '/' )
	filename = raw_filename;
      else
	filename = pathname(stream.file_name + "/" + raw_filename);
      std::cerr << "reading... rawfile=" << filename <<std::endl;
      SCIRun::Pio(stream, grid, filename);
    }
    else 
      SCIRun::Pio(stream,grid);
  }
  else { // writing
    string filename = raw_filename;
    int split = separate_raw ;
    if ( split == 1) {
      std::cerr << "SF: split \n";
      if ( filename == "" ) {
	if ( stream.file_name.c_str()) { 
	  char *tmp=strdup(stream.file_name.c_str());
	  char *dot = strrchr( tmp, '.' );
	  if (!dot ) dot = strrchr( tmp, 0);
            
	  filename = stream.file_name.substr(0,dot-tmp)+string(".raw");
	  delete tmp;
	}
	else 
	  split = 0;
      }
    }

    std::cerr << "Filename = " << filename << std::endl;
    if ( split == 1 ) {
      SCIRun::Pio(stream, split);
      SCIRun::Pio(stream, filename);
      SCIRun::Pio(stream, grid, filename );
    }
    else {
      SCIRun::Pio(stream, split);
      SCIRun::Pio(stream, grid);
    }
  }
  stream.end_class();
}


template <class T>
void ScalarFieldRGT<T>::compute_minmax()
{

  if (nx==0 || ny==0 || nz==0) return;
  double min=grid(0,0,0);
  double max=grid(0,0,0);
  for(int i=0;i<nx;i++){
    for(int j=0;j<ny;j++){
      for(int k=0;k<nz;k++){
	double val=grid(i,j,k);
	min=Min(min, val);
	max=Max(max, val);
      }
    }
  }
  data_min=min;
  data_max=max;
}


template <class T>
Vector ScalarFieldRGT<T>::gradient(const Point& p)
{
  Vector pn=p-bmin;
  double diagx=diagonal.x();
  double diagy=diagonal.y();
  double diagz=diagonal.z();
  double x=pn.x()*(nx-1)/diagx;
  double y=pn.y()*(ny-1)/diagy;
  double z=pn.z()*(nz-1)/diagz;
  int ix=(int)x;
  int iy=(int)y;
  int iz=(int)z;
  int ix1=ix+1;
  int iy1=iy+1;
  int iz1=iz+1;
  if(ix<0 || ix1>=nx)return Vector(0,0,0);
  if(iy<0 || iy1>=ny)return Vector(0,0,0);
  if(iz<0 || iz1>=nz)return Vector(0,0,0);
  double fx=x-ix;
  double fy=y-iy;
  double fz=z-iz;
  double z00=Interpolate(get_value(ix, iy, iz), get_value(ix, iy, iz1), fz);
  double z01=Interpolate(get_value(ix, iy1, iz), get_value(ix, iy1, iz1), fz);
  double z10=Interpolate(get_value(ix1, iy, iz), get_value(ix1, iy, iz1), fz);
  double z11=Interpolate(get_value(ix1, iy1, iz), get_value(ix1, iy1, iz1), fz);
  double yy0=Interpolate(z00, z01, fy);
  double yy1=Interpolate(z10, z11, fy);
  double dx=(yy1-yy0)*(nx-1)/diagonal.x();
  double x00=Interpolate(get_value(ix, iy, iz), get_value(ix1, iy, iz), fx);
  double x01=Interpolate(get_value(ix, iy, iz1), get_value(ix1, iy, iz1), fx);
  double x10=Interpolate(get_value(ix, iy1, iz), get_value(ix1, iy1, iz), fx);
  double x11=Interpolate(get_value(ix, iy1, iz1), get_value(ix1, iy1, iz1), fx);
  double y0=Interpolate(x00, x10, fy);
  double y1=Interpolate(x01, x11, fy);
  double dz=(y1-y0)*(nz-1)/diagonal.z();
  double z0=Interpolate(x00, x01, fz);
  double z1=Interpolate(x10, x11, fz);
  double dy=(z1-z0)*(ny-1)/diagonal.y();
  return Vector(dx, dy, dz);
}

template <class T>
int ScalarFieldRGT<T>::interpolate(const Point& p, double& value, int&,
				   double epsilon1, double epsilon2,
				   int) {
  return interpolate(p, value, epsilon1, epsilon2);
}

template <class T>
int ScalarFieldRGT<T>::interpolate(const Point& p, double& value, double eps,
				   double)
{

  Vector pn = p - bmin;
  const double x = pn.x()*(nx-1) / diagonal.x();
  const double y = pn.y()*(ny-1) / diagonal.y();
  const double z = pn.z()*(nz-1) / diagonal.z();
  int ix = (int)x;
  int iy = (int)y;
  int iz = (int)z;
  if (ix<0) { if (x<-eps) return 0; else ix = 0; }
  if (iy<0) { if (y<-eps) return 0; else iy = 0; }
  if (iz<0) { if (z<-eps) return 0; else iz = 0; }
  int ix1 = ix+1;
  int iy1 = iy+1;
  int iz1 = iz+1;
  if (ix1>=nx) { if (x>nx-1+eps) return 0; else ix1 = ix; }
  if (iy1>=ny) { if (y>ny-1+eps) return 0; else iy1 = iy; }
  if (iz1>=nz) { if (z>nz-1+eps) return 0; else iz1 = iz; }
  double fx = x-ix;
  double fy = y-iy;
  double fz = z-iz;
  double x00 = Interpolate(get_value(ix, iy, iz), get_value(ix1, iy, iz), fx);
  double x01 = Interpolate(get_value(ix, iy, iz1), get_value(ix1, iy, iz1), fx);
  double x10 = Interpolate(get_value(ix, iy1, iz), get_value(ix1, iy1, iz), fx);
  double x11 = Interpolate(get_value(ix, iy1, iz1), get_value(ix1, iy1, iz1), fx);
  double y0 = Interpolate(x00, x10, fy);
  double y1 = Interpolate(x01, x11, fy);
  value = Interpolate(y0, y1, fz);
  return 1;
}

template <class T>
ScalarField* ScalarFieldRGT<T>::clone()
{
  return scinew ScalarFieldRGT<T>(*this);
}

template <class T>
Vector
ScalarFieldRGT<T>::get_normal( const Point& ivoxel )
{
  return get_normal( ivoxel.x(), ivoxel.y(), ivoxel.z() );
}

template <class T>
Vector
ScalarFieldRGT<T>::get_normal( int x, int y, int z )
{
  Vector normal;

  if ( x == 0 )
    normal.x( ( get_value( x+1, y, z ) - get_value( x, y, z ) ) / 2 );
  else if ( x == nx-1 )
    normal.x( ( get_value( x, y, z ) - get_value( x-1, y, z ) ) / 2 );
  else
    normal.x( ( get_value( x-1, y, z ) + get_value( x+1, y, z ) ) / 2 );
    
  
  if ( y == 0 )
    normal.y( ( get_value( x, y+1, z ) - get_value( x, y, z ) ) / 2 );
  else if ( y == ny-1 )
    normal.y( ( get_value( x, y, z ) - get_value( x, y-1, z ) ) / 2 );
  else
    normal.y( ( get_value( x, y-1, z ) + get_value( x, y+1, z ) ) / 2 );
    
  
  if ( z == 0 )
    normal.z( ( get_value( x, y, z+1 ) - get_value( x, y, z ) ) / 2 );
  else if ( z == nz-1 )
    normal.z( ( get_value( x, y, z ) - get_value( x, y, z-1 ) ) / 2 );
  else
    normal.z( ( get_value( x, y, z-1 ) + get_value( x, y, z+1 ) ) / 2 );

  return normal;
}

template <class T>
Vector 
ScalarFieldRGT<T>::gradient(int x, int y, int z)
{
  // this tries to use central differences...

  Vector rval; // return value

  Vector h(0.5*(nx-1)/diagonal.x(),
	   0.5*(ny-1)/diagonal.y(),
	   0.5*(nz-1)/diagonal.z());
  // h is distances one over between nodes in each dimension...

  if (!x || (x == nx-1)) { // boundary...
    if (!x) {
      rval.x(2*(get_value(x+1,y,z)-get_value(x,y,z))*h.x()); // end points are rare
    } else {
      rval.x(2*(get_value(x,y,z)-get_value(x-1,y,z))*h.x());
    }
  } else { // just use central diferences...
    rval.x((get_value(x+1,y,z)-get_value(x-1,y,z))*h.x());
  }

  if (!y || (y == ny-1)) { // boundary...
    if (!y) {
      rval.y(2*(get_value(x,y+1,z)-get_value(x,y,z))*h.y()); // end points are rare
    } else {
      rval.y(2*(get_value(x,y,z)-get_value(x,y-1,z))*h.y());
    }
  } else { // just use central diferences...
    rval.y((get_value(x,y+1,z)-get_value(x,y-1,z))*h.y());
  }

  if (!z || (z == nz-1)) { // boundary...
    if (!z) {
      rval.z(2*(get_value(x,y,z+1)-get_value(x,y,z))*h.z()); // end points are rare
    } else {
      rval.z(2*(get_value(x,y,z)-get_value(x,y,z-1))*h.z());
    }
  } else { // just use central diferences...
    rval.z((get_value(x,y,z+1)-get_value(x,y,z-1))*h.z());
  }

  return rval;

}

// this stuff is for augmenting the random distributions...

template <class T>
void ScalarFieldRGT<T>::fill_gradmags() // these guys ignor the vf
{
  std::cerr << " Filling Gradient Magnitudes...\n";
  
  total_gradmag = 0.0;
  
  int nelems = (nx-1)*(ny-1)*(nz-1); // must be 3d for now...

  grad_mags.resize(nelems);

  // MAKE PARALLEL

  for(int i=0;i<nelems;i++) {
    int x,y,z;

    cell_pos(i,x,y,z); // x,y,z is lleft corner...

    Vector vs[8]; // 8 points define a cell...

    vs[0] = gradient(x,y,z);
    vs[1] = gradient(x,y+1,z);
    vs[2] = gradient(x,y,z+1);
    vs[3] = gradient(x,y+1,z+1);

    vs[4] = gradient(x+1,y,z);
    vs[5] = gradient(x+1,y+1,z);
    vs[6] = gradient(x+1,y,z+1);
    vs[7] = gradient(x+1,y+1,z+1);

    // should try different types of averages...
    // average magnitudes and directions seperately???
    // for this we just need magnitudes though.

    double ml=0;
    for(int j=0;j<8;j++) {
      ml += vs[j].length();
    }  

    grad_mags[i] = ml/8.0;;
    
    total_gradmag += grad_mags[i];
  }
}

} // end namespace FieldConverters


#endif
