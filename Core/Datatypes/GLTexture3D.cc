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

#include <Core/Datatypes/GLTexture3D.h>
#include <Core/Datatypes/Brick.h>
#include <Core/Datatypes/VolumeUtils.h>
#include <Core/Util/NotFinished.h>
#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/ThreadGroup.h>

#include <GL/gl.h>
#include <iostream>
#include <string>
#include <deque>

using std::cerr;
using std::endl;
using std::string;
using std::deque;


namespace SCIRun {

int GLTexture3D::max_workers = 0;

void glPrintError(const string& word) {
  GLenum errCode;
  const GLubyte *errString;

  if((errCode = glGetError()) != GL_NO_ERROR) {
    errString = gluErrorString(errCode);
    cerr << "OpenGL Error at " << word << ": " << errString << endl;
  }
}

static Persistent* maker()
{
    return scinew GLTexture3D;
}

PersistentTypeID GLTexture3D::type_id("GLTexture3D", "Datatype", maker);

#define GLTexture3D_VERSION 3
void GLTexture3D::io(Piostream&)
{
    NOT_FINISHED("GLTexture3D::io(Piostream&)");
}

GLTexture3D::GLTexture3D() :
  tg(0),
  texfld_(0), 
  X_(0), 
  Y_(0),
  Z_(0), 
  xmax_(0), 
  ymax_(0), 
  zmax_(0), 
  isCC_(false)
{
}


GLTexture3D::GLTexture3D(FieldHandle texfld, double &min, double &max, 
			 int use_minmax) :  
  tg(0),
  texfld_(texfld),
  X_(0), 
  Y_(0), 
  Z_(0),
  xmax_(0), 
  ymax_(0), 
  zmax_(0),
  isCC_(false) 
{
  if (texfld_->get_type_name(0) != "LatticeVol") {
    cerr << "GLTexture3D constructor error - can only make a GLTexture3D from a LatticeVol\n";
    return;
  }

  pair<double,double> minmax;
  LatVolMeshHandle mesh;
  const string type = texfld_->get_type_name(1);
  if (type == "double") {
    LatticeVol<double> *fld =
      dynamic_cast<LatticeVol<double>*>(texfld_.get_rep());
    field_minmax(*fld, minmax);
    mesh = fld->get_typed_mesh();
  } else if (type == "int") {
    LatticeVol<int> *fld =
      dynamic_cast<LatticeVol<int>*>(texfld_.get_rep());
    field_minmax(*fld, minmax);
    mesh = fld->get_typed_mesh();
  } else if (type == "short") {
    LatticeVol<short> *fld =
      dynamic_cast<LatticeVol<short>*>(texfld_.get_rep());
    field_minmax(*fld, minmax);
    mesh = fld->get_typed_mesh();
  } else if (type == "unsigned_char") {
    LatticeVol<unsigned char> *fld =
      dynamic_cast<LatticeVol<unsigned char>*>(texfld_.get_rep());
    field_minmax(*fld, minmax);
    mesh = fld->get_typed_mesh();
  } else {
    cerr << "GLTexture3D constructor error - unknown LatticeVol type: " << type << endl;
    return;
  }
  texfld_=texfld;
  xmax_=ymax_=zmax_=128;
  if( texfld_->data_at() == Field::CELL ){
    isCC_=true;
    X_ = mesh->get_nx()-1;
    Y_ = mesh->get_ny()-1;
    Z_ = mesh->get_nz()-1;
  } else {
    isCC_=false;
    X_ = mesh->get_nx();
    Y_ = mesh->get_ny();
    Z_ = mesh->get_nz();
  }    


  minP_ = mesh->get_min();
  maxP_ = mesh->get_max();
  cerr <<"X_, Y_, Z_ = "<<X_<<", "<<Y_<<", "<<Z_<<endl;
  cerr << "use_minmax = "<<use_minmax<<"  min="<<min<<" max="<<max<<"\n";
  cerr << "    fieldminmax: min="<<minmax.first<<" max="<<minmax.second<<"\n";
  if (use_minmax) {
    min_ = min;
    max_ = max;
  } else {
    min = min_ = minmax.first;
    max = max_ = minmax.second;
  }
  cerr << "    texture: min="<<min<<"  max="<<max<<"\n";
  set_bounds();
  compute_tree_depth(); 
  build_texture();
}

GLTexture3D::~GLTexture3D()
{
  delete bontree_;
}






void GLTexture3D::set_bounds()
{
  Vector diag = maxP_ - minP_;
  std::cerr<<"Bounds = "<<minP_<<", "<<maxP_<<std::endl;

  dx_ = diag.x()/(X_-1);
  dy_ = diag.y()/(Y_-1);
  dz_ = diag.z()/(Z_-1); 
}

void GLTexture3D::build_texture()
{
  max_workers = Max(Thread::numProcessors()/2, 8);
  Semaphore* thread_sema = scinew Semaphore( "worker count semhpore",
					  max_workers);  

  string  group_name =  "thread group ";
  group_name = group_name + "0";
  tg = scinew ThreadGroup( group_name.c_str());

  string type = texfld_->get_type_name(1);
  cerr << "Type = " << type << endl;
  
  if (type == "double") {
    bontree_ = build_bon_tree(minP_, maxP_, 0, 0, 0, X_, Y_, Z_, 0, 
	       dynamic_cast<LatticeVol<double>*>(texfld_.get_rep()), 0, 
			   thread_sema, tg);
  } else if (type == "int") {
    bontree_ = build_bon_tree(minP_, maxP_, 0, 0, 0, X_, Y_, Z_, 0, 
	       dynamic_cast<LatticeVol<int>*>(texfld_.get_rep()), 0, 
			   thread_sema, tg);
  } else if (type == "short") {
    bontree_ = build_bon_tree(minP_, maxP_, 0, 0, 0, X_, Y_, Z_, 0, 
	       dynamic_cast<LatticeVol<short>*>(texfld_.get_rep()), 0, 
			   thread_sema, tg);
  } else if (type == "unsigned_char") {
    bontree_ = build_bon_tree(minP_, maxP_, 0, 0, 0, X_, Y_, Z_, 0, 
	       dynamic_cast<LatticeVol<unsigned char>*>(texfld_.get_rep()), 0, 
			   thread_sema, tg);
  } else {
    cerr<<"Error: cast didn't work!\n";
  }

  tg->join();
  delete tg;
  thread_sema->down(max_workers);
  ASSERT(bontree_ != 0x0);
}


template<> 
bool GLTexture3D::get_dimensions(LatVolMeshHandle m,
				 int& nx, int& ny, int& nz)
{
  nx = m->get_nx();
  ny = m->get_ny();
  nz = m->get_nz();
  return true;
}

bool
GLTexture3D::get_dimensions( int& nx, int& ny, int& nz)
{
  LatVolMesh *meshpointer = 
    dynamic_cast<LatVolMesh *>(texfld_->mesh().get_rep());
  if (meshpointer)
  {
    LatVolMeshHandle mesh(meshpointer);
    return get_dimensions(mesh, nx, ny, nz);
  }
  else
  {
    cerr << "GLTexture3D constructor error - no mesh in field." << endl;
    return false;
  }
}


bool
GLTexture3D::set_brick_size(int bsize)
{
  xmax_ = ymax_ = zmax_ = bsize;
  int x,y,z;
  if (get_dimensions( x, y, z) ){
    if( texfld_->data_at() == Field::CELL ){
      isCC_=true;
      X_ = x-1;
      Y_ = y-1;
      Z_ = z-1;
    } else {
      isCC_=false;
      X_ = x;
      Y_ = y;
      Z_ = z;
    }    
    if( bontree_ ) delete bontree_;
    compute_tree_depth();
    build_texture();
    return true;
  } else {
    return false;
  }
}

void
GLTexture3D::set_field( FieldHandle texfld )
{
  if (texfld_->get_type_name(0) != "LatticeVol") {
    cerr << "GLTexture3D constructor error - can only make a GLTexture3D from a LatticeVol\n";
    return;
  }
  texfld_=texfld;

  int size = std::max(X_,Y_);
  size = std::max(size,Z_);

  xmax_ = ymax_ = zmax_ = 128;

  set_bounds();
  compute_tree_depth(); 
  build_texture();
} 
 
void GLTexture3D::compute_tree_depth()
{
  int xdepth = 0, ydepth = 0, zdepth = 0;
  int x = xmax_, y = ymax_, z = zmax_;

  while(x < X_){
    x = (x * 2) - 1;
    xdepth++;
  }
  while(y < Y_){
    y = (y * 2) - 1;
    ydepth++;
  }
  while(z < Z_){
    z = (z * 2) - 1;
    zdepth++;
  }

  levels_ = (( xdepth > ydepth)? ((xdepth > zdepth)? xdepth:
				    ((ydepth > zdepth)? ydepth:zdepth)):
	       ( ydepth > zdepth)? ydepth : zdepth);
}


bool GLTexture3D::set_max_brick_size(int maxBrick)
{


   GLint xtex = 0, ytex = 0, ztex = 0; 
   int x,y,z;
   x = y = z = maxBrick;
   glEnable(GL_TEXTURE_3D);
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   


   glPrintError("glEnable(GL_TEXTURE_3D)");
   //glTexImage3D(GL_PROXY_TEXTURE_3D, 0, GL_INTENSITY8_EXT, x, y, z, 0, 
   //                   GL_RED, GL_UNSIGNED_BYTE, NULL); 
   glTexImage3D(GL_PROXY_TEXTURE_3D, 0, GL_INTENSITY8, x, y, z, 0, 
                      GL_RED, GL_UNSIGNED_BYTE, NULL); 
   glPrintError("glTexImage3DEXT");
   glGetTexLevelParameteriv( GL_PROXY_TEXTURE_3D, 0, 
                             GL_TEXTURE_WIDTH, &xtex); 
   glPrintError("glGetTexLevelParameteriv1");
   glGetTexLevelParameteriv( GL_PROXY_TEXTURE_3D, 0, 
                             GL_TEXTURE_HEIGHT, &ytex); 
   glPrintError("glGetTexLevelParameteriv2");
   glGetTexLevelParameteriv( GL_PROXY_TEXTURE_3D, 0, 
                             GL_TEXTURE_DEPTH, &ztex); 
   glPrintError("glGetTexLevelParameteriv3");
   glDisable(GL_TEXTURE_3D);
   glPrintError("glDisable(GL_TEXTURE_3D)");   
   if( xtex && ytex) { // we can accommodate 
     xmax_ = ymax_ = zmax_ = maxBrick; 
     return true;
   } else {
     return false;
   }
}

double
GLTexture3D::SETVAL(double val)
{
  double v = (val - min_)*255/(max_ - min_);
  if ( v < 0 ) return 0;
  else if (v > 255) return 255;
  else return v;
}

unsigned char
GLTexture3D::SETVALC(double val)
{
  return (unsigned char)SETVAL(val);
}


GLTexture3D::run_make_low_res_brick_data
::run_make_low_res_brick_data(GLTexture3D* tex3D,
			      Semaphore *thread,
			      int xmax, int ymax, int zmax,
			      int xsize, int ysize, int zsize,
			      int xoff, int yoff, int zoff,
			      int& padx, int& pady, int& padz,
			      int level, Octree<Brick*>* node,
			      Array3<unsigned char>*& bd) :
  tex3D_(tex3D),
  parent_(node), 
  thread_sema_( thread ), 
  xmax_(xmax), 
  ymax_(ymax), 
  zmax_(zmax),
  xsize_(xsize), 
  ysize_(ysize), 
  zsize_(zsize),
  xoff_(xoff), 
  yoff_(yoff), 
  zoff_(zoff),
  padx_(padx), 
  pady_(pady), 
  padz_(padz),
  level_(level), 
  bd_(bd)
{
  // constructor
}

void
GLTexture3D::run_make_low_res_brick_data::run() 
{
  using SCIRun::Interpolate;

  int ii,jj,kk;
  Brick *brick = 0;
  Array3<unsigned char>* brickTexture;

//   if( level == 0 ){
//     double  i,j,k;
//     int k1,j1,i1;
//     double dk,dj,di, k00,k01,k10,k11,j00,j01;
//     bool iswitch = false , jswitch = false, kswitch = false;
//     dx = (double)(xsize_-1)/(xmax_-1.0);
//     dy = (double)(ysize_-1)/(ymax_-1.0);
//     dz = (double)(zsize_-1)/(zmax_-1.0);
//     int x,y,z;
//     for( kk = 0, k = 0; kk < zmax_; kk++, k+=dz){
//       if ( dz*kk >= zmax_ )  z = 1; else z = 0;
//       if (!kswitch)
// 	if ( dz*kk >= zmax_ ){ k = zmax_ - dz*kk + 1; kswitch = true; }
//       k1 = ((int)k + 1 >= zmax_)?(int)k:(int)k + 1;
//       if(k1 == (int)k ) { dk = 0; } else {dk = k1 - k;}
//       for( jj = 0, j = 0; jj < ymax_; jj++, j+=dy){
// 	if( dy*jj >= ymax_) y = 2; else y = 0;
// 	if( !jswitch )
// 	  if( dy*jj >= ymax_) { j = ymax_ - dy*jj + 1; jswitch = true; }
// 	j1 = ((int)j + 1 >= ymax_)?(int)j:(int)j + 1 ;
// 	if(j1 == (int)j) {dj = 0;} else { dj = j1 - j;} 
// 	for (ii = 0, i = 0; ii < xmax_; ii++, i+=dx){
// 	  if( dx*ii >= xmax_ ) x = 4; else x = 0;
// 	  if( !iswitch )
// 	    if( dx*ii >= xmax_ ) { i = xmax_ - dz*ii + 1; iswitch = true; }
// 	  i1 = ((int)i + 1 >= xmax_)?(int)i:(int)i + 1 ;
// 	  if( i1 == (int)i){ di = 0;} else {di = i1 - i;}

// 	  brick = (*((*this->parent_)[x+y+z]))();
// 	  if( brick == 0 ){
// 	    (*bd)(kk,jj,ii) = (unsigned char)0;
// 	  } else {
// 	    brickTexture = brick->texture();
// 	    k00 = Interpolate(tex3D_->SETVALC( (*brickTexture)(i,j,k) ),
// 			      tex3D_->SETVALC( (*brickTexture)(i,j,k1)),dk);
// 	    k01 = Interpolate(tex3D_->SETVALC( (*brickTexture)(i1,j,k)),
// 			      tex3D_->SETVALC( (*brickTexture)(i1,j,k1)),dk);
// 	    k10 = Interpolate(tex3D_->SETVALC( (*brickTexture)(i,j1,k)),
// 			      tex3D_->SETVALC( (*brickTexture)(i,j,k1)),dk);
// 	    k11 = Interpolate(tex3D_->SETVALC( (*brickTexture)(i1,j1,k)),
// 			      tex3D_->SETVALC( (*brickTexture)(i1,j1,k1)),dk);
// 	    j00 = Interpolate(k00,k10,dj);
// 	    j01 = Interpolate(k01,k11,dj);
// 	    (*bd_)(kk,jj,ii) = (unsigned char)Interpolate(j00,j01,di);
// 	  }
// 	}
//       }
//     }
// //    thread_sema_->up();
//     return;
//   } else {
    int  i,j,k;
    int x,y,z;
    for( kk = 0, k = 0; kk < zmax_; kk++, k+=2){
      if ( 2*kk >= zmax_ )  z = 1; else z = 0;
      if ( 2*kk == zmax_ ) k = 1;
      for( jj = 0, j = 0; jj < ymax_; jj++, j+=2){
	if( 2*jj >= ymax_) y = 2; else y = 0;
	if( 2*jj == ymax_) j = 1;
	for (ii = 0, i = 0; ii < xmax_; ii++, i+=2){
	  if( 2*ii >= xmax_ ) x = 4; else x = 0;
	  if( 2*ii == xmax_ ) i = 1;

	  brick = (*((*this->parent_)[x+y+z]))();
	  brickTexture = brick->texture();

	  // This code does simple subsampling.  Uncomment the 
	  // center section to perform averaging.
	  if( brick == 0 ){
	    (*bd_)(kk,jj,ii) = (unsigned char)0;
//////////// Uncomment for texel averageing
// 	  } else if((ii > 0 && ii < xmax_ - 1) &&
// 	     (jj > 0 && jj < ymax_ - 1) &&
// 	     (kk > 0 && kk < zmax_ - 1)){
// 	    (*bd_)(kk,jj,ii) = (0.5*(*brickTexture)(k,j,i)           +
// 			       0.083333333*(*brickTexture)(k,j,i-1) +
// 			       0.083333333*(*brickTexture)(k,j,i+1) +
// 			       0.083333333*(*brickTexture)(k,j-1,i) +
// 	                       0.083333333*(*brickTexture)(k,j+1,i) +
// 			       0.083333333*(*brickTexture)(k-1,j,i) +
// 			       0.083333333*(*brickTexture)(k+1,j,i));
///////////
	  } else {
	    // texel subsampling--always select border cells.
	    // leave uncommented even if averaging is uncommented.
	    (*bd_)(kk,jj,ii) = (*brickTexture)(k,j,i);
	  }
	}
      }
    }
    thread_sema_->up();
//  }    
}

} // End namespace SCIRun

