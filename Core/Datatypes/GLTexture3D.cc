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
#include <Core/Containers/String.h>
#include <Core/Datatypes/LatticeVol.h>

#include <GL/gl.h>
#include <iostream>
#include <string>
#include <deque>

using std::cerr;
using std::endl;
using std::string;
using std::deque;


namespace SCIRun {

void glPrintError(const string& word){
  GLenum errCode;
  const GLubyte *errString;

  if((errCode = glGetError()) != GL_NO_ERROR) {
    errString = gluErrorString(errCode);
    cerr<<"OpenGL Error at "<<word.c_str()<<": "<< errString<<endl;
  }
}

static Persistent* maker()
{
    return scinew GLTexture3D;
}

PersistentTypeID GLTexture3D::type_id("GLTexture3D", "Datatype"
, maker);
#define GLTexture3D_VERSION 3
void GLTexture3D::io(Piostream&)
{
    NOT_FINISHED("GLTexture3D::io(Piostream&)");
}

GLTexture3D::GLTexture3D() :
  texfld_(0), mesh_(0), X_(0), Y_(0),
  Z_(0), xmax_(0), ymax_(0), zmax_(0), isCC_(false)
{
}

GLTexture3D::GLTexture3D(FieldHandle texfld) :
  texfld_(0), mesh_(0), X_(0), Y_(0),
  Z_(0), xmax_(0), ymax_(0), zmax_(0), isCC_(false)
{
  if (texfld_->get_type_name(0) != "LatticeVol") {
    cerr << "GLTexture3D constructor error - can only make a GLTexture3D from a LatticeVol\n";
    return;
  }

  pair<double,double> minmax;

  string type = texfld_->get_type_name(1);
  if (type == "double") {
    LatticeVol<double> *fld =
      dynamic_cast<LatticeVol<double>*>(texfld_.get_rep());
    field_minmax(*fld, minmax);
    mesh_ = fld->get_typed_mesh();
  } else if (type == "float") {
    LatticeVol<float> *fld =
      dynamic_cast<LatticeVol<float>*>(texfld_.get_rep());
    field_minmax(*fld, minmax);
    mesh_ = fld->get_typed_mesh();
  } else if (type == "unsigned int") {
    LatticeVol<unsigned int> *fld =
      dynamic_cast<LatticeVol<unsigned int>*>(texfld_.get_rep());
    field_minmax(*fld, minmax);
    mesh_ = fld->get_typed_mesh();
  } else if (type == "int") {
    LatticeVol<int> *fld =
      dynamic_cast<LatticeVol<int>*>(texfld_.get_rep());
    field_minmax(*fld, minmax);
    mesh_ = fld->get_typed_mesh();
  } else if (type == "unsigned short") {
    LatticeVol<unsigned short> *fld =
      dynamic_cast<LatticeVol<unsigned short>*>(texfld_.get_rep());
    field_minmax(*fld, minmax);
    mesh_ = fld->get_typed_mesh();
  } else if (type == "short") {
    LatticeVol<short> *fld =
      dynamic_cast<LatticeVol<short>*>(texfld_.get_rep());
    field_minmax(*fld, minmax);
    mesh_ = fld->get_typed_mesh();
  } else if (type == "unsigned char") {
    LatticeVol<unsigned char> *fld =
      dynamic_cast<LatticeVol<unsigned char>*>(texfld_.get_rep());
    field_minmax(*fld, minmax);
    mesh_ = fld->get_typed_mesh();
  } else if (type == "char") {
    LatticeVol<char> *fld =
      dynamic_cast<LatticeVol<char>*>(texfld_.get_rep());
    field_minmax(*fld, minmax);
    mesh_ = fld->get_typed_mesh();
  } else {
    cerr << "GLTexture3D constructor error - unknown LatticeVol type: " << type << endl;
    return;
  }
  texfld_=texfld;
  xmax_=ymax_=zmax_=128;
  isCC_=false;
  X_ = mesh_->get_nx();
  Y_ = mesh_->get_ny();
  Z_ = mesh_->get_nz();
  minP_ = mesh_->get_min();
  maxP_ = mesh_->get_max();
  min_ = minmax.first;
  max_ = minmax.second;
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
  string type = texfld_->get_type_name(1);
  cerr << "Type = " << type << endl;

  if (type == "double") {
    bontree_ = build_bon_tree(minP_, maxP_, 0, 0, 0, X_, Y_, Z_, 0, 
	       dynamic_cast<LatticeVol<double>*>(texfld_.get_rep()), 0);
  } else if (type == "float") {
    bontree_ = build_bon_tree(minP_, maxP_, 0, 0, 0, X_, Y_, Z_, 0, 
	       dynamic_cast<LatticeVol<float>*>(texfld_.get_rep()), 0);
  } else if (type == "unsigned int") {
    bontree_ = build_bon_tree(minP_, maxP_, 0, 0, 0, X_, Y_, Z_, 0, 
	       dynamic_cast<LatticeVol<unsigned int>*>(texfld_.get_rep()), 0);
  } else if (type == "int") {
    bontree_ = build_bon_tree(minP_, maxP_, 0, 0, 0, X_, Y_, Z_, 0, 
	       dynamic_cast<LatticeVol<int>*>(texfld_.get_rep()), 0);
  } else if (type == "unsigned short") {
    bontree_ = build_bon_tree(minP_, maxP_, 0, 0, 0, X_, Y_, Z_, 0, 
	       dynamic_cast<LatticeVol<unsigned short>*>(texfld_.get_rep()),0);
  } else if (type == "short") {
    bontree_ = build_bon_tree(minP_, maxP_, 0, 0, 0, X_, Y_, Z_, 0, 
	       dynamic_cast<LatticeVol<short>*>(texfld_.get_rep()), 0);
  } else if (type == "unsigned char") {
    bontree_ = build_bon_tree(minP_, maxP_, 0, 0, 0, X_, Y_, Z_, 0, 
	       dynamic_cast<LatticeVol<unsigned char>*>(texfld_.get_rep()), 0);
  } else if (type == "char") {
    bontree_ = build_bon_tree(minP_, maxP_, 0, 0, 0, X_, Y_, Z_, 0, 
	       dynamic_cast<LatticeVol<char>*>(texfld_.get_rep()), 0);
  } else {
    cerr<<"Error: cast didn't work!\n";
  }
  ASSERT(bontree_ != 0x0);
}

bool
GLTexture3D::set_brick_size(int bsize)
{
  xmax_ = ymax_ = zmax_ = bsize;
  X_ = mesh_->get_nx();
  Y_ = mesh_->get_ny();
  Z_ = mesh_->get_nz();
  
  if( bontree_ ) delete bontree_;
  compute_tree_depth();
  build_texture();
  return true;
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

template <class T>
Octree<Brick*>*
GLTexture3D::build_bon_tree(Point min, Point max,
			    int xoff, int yoff, int zoff,
			    int xsize, int ysize, int zsize,
			    int level, T *tex, Octree<Brick*>* parent)
{

    /* The cube is numbered in the following way 
     
          2________6        y
         /|        |        |  
        / |       /|        |
       /  |      / |        |
      /   0_____/__4        |
     3---------7   /        |_________ x
     |  /      |  /         /
     | /       | /         /
     |/        |/         /
     1_________5         /
                        z  
  */

  Octree<Brick *> *node;

  if (xoff > X_ || yoff > Y_ || zoff> Z_){
    node = 0;
    //return node;
  }

  Brick* brick;
  Array3<unsigned char> *brickData;
  // Check to make sure that we can accommodate the requested texture
  GLint xtex =0 , ytex = 0 , ztex = 0;

  if ( xsize <= xmax_ ) xtex = 1;
  if ( ysize <= ymax_ ) ytex = 1;
  if ( zsize <= zmax_ ) ztex = 1;

  brickData = scinew Array3<unsigned char>();
  int padx = 0,pady = 0,padz = 0;

  if( xtex && ytex && ztex) { // we can accommodate
    int newx = xsize, newy = ysize, newz = zsize;
    if (xsize < xmax_){
      padx = xmax_ - xsize;
      newx = xmax_;
    }
    if (ysize < ymax_){
      pady = ymax_ - ysize;
      newy = ymax_;
    }
    if (zsize < zmax_){
      padz = zmax_ - zsize;
      newz = zmax_;
    }

    make_brick_data(newx,newy,newz,xsize,ysize,zsize, xoff,yoff,zoff,
		    tex, brickData);

    brick = scinew Brick(min, max, padx, pady, padz, level, brickData);

    node = scinew Octree<Brick*>(brick, Octree<Brick *>::LEAF, parent );
  } else { // we must subdivide

    make_low_res_brick_data(xmax_, ymax_, zmax_, xsize, ysize, zsize,
			    xoff, yoff, zoff, level, padx, pady, padz,
			    tex, brickData);

    brick = scinew Brick(min, max, padx, pady, padz, level, brickData);

    node = scinew Octree<Brick*>(brick, Octree<Brick *>::PARENT, parent);

    int sx = xmax_, sy = ymax_, sz = zmax_, tmp;
    tmp = xmax_;
    while( tmp < xsize){
      sx = tmp;
      tmp = tmp*2 -1;
    }
    tmp = ymax_;
    while( tmp < ysize){
      sy = tmp;
      tmp = tmp*2 -1;
    }
    tmp = zmax_;
    while( tmp < zsize){
      sz = tmp;
      tmp = tmp*2 -1;
    }   
 
    level++;

    int X2, Y2, Z2;
    X2 = largestPowerOf2( xsize -1);
    Y2 = largestPowerOf2( ysize -1);
    Z2 = largestPowerOf2( zsize -1);


    Vector diag = max - min;
    Point mid;
    if( Z2 == Y2 && Y2 == X2 ){mid = min + Vector(dx_ * (sx-1), dy_ * (sy-1),
						  dz_ * (sz-1));
      for(int i = 0; i < 8; i++){
	build_child(i, min, mid, max, xoff, yoff, zoff,
		    xsize, ysize, zsize, sx, sy, sz,level,tex, node);
      }
    } else if( Z2 > Y2 && Z2 > X2 ) {
      mid = min + Vector(diag.x(),
			  diag.y(),
			  dz_ * (sz-1));
      
      build_child(0, min, mid, max, xoff, yoff, zoff,
		  xsize, ysize, zsize, xsize, ysize, sz, level, tex, node);
      build_child(1, min, mid, max, xoff, yoff, zoff,
		  xsize, ysize, zsize, xsize, ysize, sz, level, tex, node);
    } else  if( Y2 > Z2 && Y2 > X2 ) {
      mid = min + Vector(diag.x(),
			 dy_ * (sy - 1),
			 diag.z());
      build_child(0, min, mid, max, xoff, yoff, zoff,
		  xsize, ysize, zsize, xsize, sy, zsize, level, tex, node);
      build_child(2, min, mid, max, xoff, yoff, zoff,
		  xsize, ysize, zsize, xsize, sy, zsize, level, tex, node);
    } else  if( X2 > Z2 && X2 > Y2 ) {
      mid = min + Vector(dx_ * (sx-1),
			 diag.y(),
			 diag.z());
      build_child(0, min, mid, max, xoff, yoff, zoff,
		  xsize, ysize, zsize, sx, ysize, zsize, level, tex, node);
      build_child(4,min, mid, max, xoff, yoff, zoff,
		  xsize, ysize, zsize, sx, ysize, zsize, level, tex, node);
    } else if( Z2 == Y2 ){
      mid = min + Vector(diag.x(),
			 dy_ * (sy - 1),
			 dz_ * (sz - 1));
      build_child(0,min, mid, max, xoff, yoff, zoff,
		  xsize, ysize, zsize, xsize, sy, sz, level, tex, node);
      build_child(1,min, mid, max, xoff, yoff, zoff,
		  xsize, ysize, zsize, xsize, sy, sz, level, tex, node);
      build_child(2,min, mid, max, xoff, yoff, zoff,
		  xsize, ysize, zsize, xsize, sy, sz, level, tex, node);
      build_child(3,min, mid, max, xoff, yoff, zoff,
		  xsize, ysize, zsize, xsize, sy, sz, level, tex, node);
    } else if( X2 == Y2 ){
      mid = min + Vector(dx_ * (sx - 1), dy_ * (sy-1),
			 diag.z());
      build_child(0,min, mid, max, xoff, yoff, zoff,
		  xsize, ysize, zsize, sx, sy, zsize, level, tex, node);
      build_child(2,min, mid, max, xoff, yoff, zoff,
		  xsize, ysize, zsize, sx, sy, zsize, level, tex, node);
      build_child(4,min, mid, max, xoff, yoff, zoff,
		  xsize, ysize, zsize, sx, sy, zsize, level, tex, node);
      build_child(6,min, mid, max, xoff, yoff, zoff,
		  xsize, ysize, zsize, sx, sy, zsize, level, tex, node);
    } else if( Z2 == X2 ){
      mid = min + Vector(dx_ * (sx-1),
			 diag.y(),
			 dz_ * (sz-1));
      build_child(0,min, mid, max, xoff, yoff, zoff,
		  xsize, ysize, zsize, sx, ysize, sz, level, tex, node);
      build_child(1,min, mid, max, xoff, yoff, zoff,
		  xsize, ysize, zsize, sx, ysize, sz, level, tex, node);
      build_child(4,min, mid, max, xoff, yoff, zoff,
		  xsize, ysize, zsize, sx, ysize, sz, level, tex, node);
      build_child(5,min, mid, max, xoff, yoff, zoff,
		  xsize, ysize, zsize, sx, ysize, sz, level, tex, node);
    }
  }
  return node;
}

template <class T>
void GLTexture3D::build_child(int i, Point min, Point mid, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level,  T* tex, Octree<Brick*>* node)
{
  Point pmin, pmax;

  switch( i ) {
  case 0:
    pmin = min;
    pmax = mid;
    node->SetChild(0, build_bon_tree(pmin, pmax, xoff, yoff, zoff,
				   X2, Y2, Z2, level, tex, node));
    break;
  case 1:
    pmin = min;
    pmax = mid;
    pmin.z(mid.z());
    pmax.z(max.z());
    node->SetChild(1, build_bon_tree(pmin, pmax,
				   xoff, yoff, zoff + Z2 -1,
				   X2, Y2, zsize-Z2+1, level, tex, node));
    break;
  case 2:
    pmin = min;
    pmax = mid;
    pmin.y(mid.y());
    pmax.y(max.y());
    node->SetChild(2, build_bon_tree(pmin, pmax,
				   xoff, yoff + Y2 - 1, zoff,
				   X2, ysize - Y2 + 1, Z2, level, tex, node));
    break;
  case 3:
    pmin = mid;
    pmax = max;
    pmin.x(min.x());
    pmax.x(mid.x());
    node->SetChild(3, build_bon_tree(pmin, pmax,
				   xoff, yoff + Y2 - 1 , zoff + Z2 - 1,
				   X2, ysize - Y2 + 1, zsize - Z2 + 1, level, tex, node));
    break;
  case 4:
    pmin = min;
    pmax = mid;
    pmin.x(mid.x());
    pmax.x(max.x());
    node->SetChild(4, build_bon_tree(pmin, pmax,
				   xoff + X2 - 1, yoff, zoff,
				   xsize - X2 + 1, Y2, Z2, level, tex, node));
    break;
  case 5:
    pmin = mid;
    pmax = max;
    pmin.y(min.y());
    pmax.y(mid.y());
    node->SetChild(5, build_bon_tree(pmin, pmax,
				   xoff + X2 - 1, yoff, zoff +  Z2 - 1,
				   xsize - X2 + 1, Y2, zsize - Z2 + 1, level, tex, node));
    break;
  case 6:
    pmin = mid;
    pmax = max;
    pmin.z(min.z());
    pmax.z(mid.z());
    node->SetChild(6, build_bon_tree(pmin, pmax,
				   xoff + X2 - 1, yoff + Y2 - 1, zoff,
				   xsize - X2 + 1, ysize - Y2 + 1, Z2, level, tex, node));
    break;
  case 7:
   pmin = mid;
   pmax = max;
   node->SetChild(7, build_bon_tree(pmin, pmax,  xoff + X2 - 1,
				  yoff + Y2 - 1, zoff +  Z2 - 1,
				  xsize - X2 + 1, ysize - Y2 + 1,
				  zsize - Z2 + 1, level, tex, node));
   break;
  default:
    break;
  }
}

double
GLTexture3D::SETVAL(double val) {
  double v = (val - min_)*255/(max_ - min_);
  if ( v < 0 ) return 0;
  else if (v > 255) return 255;
  else return v;
}

template <class T>
void GLTexture3D::make_brick_data(int newx, int newy, int newz,
				int xsize, int ysize, int zsize,
				int xoff, int yoff, int zoff, T *tex,
				Array3<unsigned char>*& bd)
{
  int i,j,k,ii,jj,kk;

  bd->newsize( newz, newy, newx);
  for(kk = 0, k = zoff; kk < zsize; kk++, k++)
    for(jj = 0, j = yoff; jj < ysize; jj++, j++)
      for(ii = 0, i = xoff; ii < xsize; ii++, i++){
	(*bd)(kk,jj,ii) = SETVAL( tex->fdata()(i,j,k) );
  }

}

template <class T>						
void GLTexture3D::make_low_res_brick_data(int xmax, int ymax, int zmax,
				      int xsize, int ysize, int zsize,
				      int xoff, int yoff, int zoff,
				      int level, int& padx, int& pady,
				      int& padz, T* tex,
				      Array3<unsigned char>*& bd)
{

  double  i,j,k;
  int ii,jj,kk;
  double dx, dy, dz;
  bd->newsize( zmax, ymax, xmax);

  if( level == 0 ){
    dx = (double)(xsize-1)/(xmax-1.0);
    dy = (double)(ysize-1)/(ymax-1.0);
    dz = (double)(zsize-1)/(zmax-1.0);

    int k1,j1,i1;
    double dk,dj,di, k00,k01,k10,k11,j00,j01;

    for(kk = 0, k = zoff; kk < zmax; kk++, k+=dz){
      k1 = ((int)k + 1 >= zoff+zsize-1)?k:(int)k + 1 ;
      if(k1 == (int)k ) { dk = 0; } else {dk = k1 - k;}
      for(jj = 0,j = yoff; jj < ymax; jj++, j+=dy){
	j1 = ((int)j + 1 >= yoff+ysize-1)?j:(int)j + 1 ;
	if(j1 == (int)j) {dj = 0;} else { dj = j1 - j;} 
	for(ii = 0, i = xoff; ii < xmax; ii++, i+=dx){
	  i1 = ((int)i + 1 >= xoff+xsize-1)?i:(int)i + 1 ;
	  if( i1 == (int)i){ di = 0;} else {di = i1 - i;}
	  k00 = Interpolate(SETVAL( tex->fdata()(i,j,k) ),
			    SETVAL( tex->fdata()(i,j,k1)),dk);
	  k01 = Interpolate(SETVAL( tex->fdata()(i1,j,k)),
			    SETVAL( tex->fdata()(i1,j,k1)),dk);
	  k10 = Interpolate(SETVAL( tex->fdata()(i,j1,k)),
			    SETVAL( tex->fdata()(i,j,k1)),dk);
	  k11 = Interpolate(SETVAL( tex->fdata()(i1,j1,k)),
			    SETVAL( tex->fdata()(i1,j1,k1)),dk);
	  j00 = Interpolate(k00,k10,dj);
	  j01 = Interpolate(k01,k11,dj);
	  (*bd)(kk,jj,ii) = Interpolate(j00,j01,di);
	}
      }
    }
  } else {


    if( xmax > xsize ) {
      dx = 1; padx=(xmax - xsize);
    } else {
      dx = pow(2.0, levels_ - level);
      if( xmax * dx > xsize){
	padx = (xmax*dx - xsize)/dx;
      }
    }
    if( ymax > ysize ) {
      dy = 1; pady = (ymax - ysize);
    } else {
      dy = pow(2.0, levels_ - level);
      if( ymax * dy > ysize){
	pady = (ymax*dy - ysize)/dy;
      }
    }
    if( zmax > zsize ) {
      dz = 1; padz = (zmax - zsize);
    } else {
      dz = pow(2.0, levels_ - level);
      if( zmax * dz > zsize){
	padz = (zmax*dz - zsize)/dz;
      }
    }
/*     if(debug){ */
/*       cerr<<"xmax = "<< xmax * dx<<", xsize = "<<xsize<<endl; */
/*       cerr<<"ymax = "<< ymax * dy<<", ysize = "<<ysize<<endl; */
/*       cerr<<"zmax = "<< zmax * dz<<", zsize = "<<zsize<<endl; */
/*       cerr<<"dx, dy, dz = "<< dx<<", "<<dy<<", "<<dz<<endl; */
    /*       cerr<<"padx, pady, padz = "<< padx<<", "<<pady<<", "<<padz<<endl; */
  }
  
  for(kk = 0, k = zoff; kk < zmax; kk++, k+=dz){
      for(jj = 0, j = yoff; jj < ymax; jj++, j+=dy){
	for(ii = 0, i = xoff; ii < xmax; ii++, i+=dx){
	  if( i < xoff + xsize && j < yoff + ysize && k < zoff + zsize){
	    (*bd)(kk,jj,ii) = SETVAL( tex->fdata()(i,j,k) );
	  }
	}
      }
  }    
}


} // End namespace SCIRun

