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
//#include <Core/Datatypes/ScalarFieldRG.h>
//#include <Uintah/Datatypes/NCScalarField.h>
//#include <Uintah/Datatypes/CCScalarField.h>


#include <GL/gl.h>
#include <iostream>
#include <string>
#include <deque>

using std::cerr;
using std::endl;
using std::string;
using std::deque;


namespace SCIRun {



// NCScalarField<double> sfdr0;
// NCScalarField<int> sfir0; 
// NCScalarField<long> sflr0; 
// CCScalarField<double> sfdr1; 
// CCScalarField<int> sfir1; 
// CCScalarField<long> sflr1; 


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
  _tex(0), X(0), Y(0),
  Z(0), xmax(0), ymax(0), zmax(0), isCC(false)
{
}

GLTexture3D::GLTexture3D(void /*ScalarFieldRGBase*/ *tex ) :
  /*_tex(tex), X(tex->nx), Y(tex->ny),
    Z(tex->nz),  */
  xmax(128), ymax(128), zmax(128), isCC(false)
{
  /*  tex->get_bounds( minP, maxP );
      tex->get_minmax( _min, _max );*/
  SetBounds();
  computeTreeDepth(); 
//   bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, tex, 0);
  BuildTexture();
}

GLTexture3D::~GLTexture3D()
{
  delete bontree;
}

void GLTexture3D::SetBounds()
{
  Vector diag = maxP - minP;
  std::cerr<<"Bounds = "<<minP<<", "<<maxP<<std::endl;

  dx = diag.x()/(X-1);
  dy = diag.y()/(Y-1);
  dz = diag.z()/(Z-1); 
}

void GLTexture3D::BuildTexture()
{
#if 0
  cerr<<"Type = "<<_tex->getType()<<endl;
  if( _tex->getRGDouble() ){
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,
			   _tex->getRGDouble(), 0);
  } else if( _tex->getRGFloat() ){
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,
			   _tex->getRGFloat(), 0);
  } else if( _tex->getRGInt() ) {
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,
			   _tex->getRGInt(), 0);
  } else if( _tex->getRGShort() ){
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,
			   _tex->getRGShort(), 0);
  } else if( _tex->getRGUchar() ){
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,
			   _tex->getRGUchar(), 0);
  } else if( _tex->getRGChar() ){
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,
			   _tex->getRGChar(), 0);
  } /*else if(NCScalarField<double> *sfd =
	    dynamic_cast<NCScalarField<double> *> (_tex)){
    cerr<<"Type = <NCScalarField<double>\n";
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, sfd, 0);
  } else if(NCScalarField<int> *sfi =
	    dynamic_cast<NCScalarField<int> *> (_tex)){
    cerr<<"Type = NCScalarField<int>\n";
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, sfi, 0);
  } else if(NCScalarField<long> *sfl =
	    dynamic_cast<NCScalarField<long> *> (_tex)){
    cerr<<"Type = NCScalarField<long>\n";
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, sfl, 0);
  } else if(CCScalarField<double> *sfd =
	    dynamic_cast<CCScalarField<double> *> (_tex)){
    cerr<<"Type = CCScalarField<double>\n";
    isCC = true;
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, sfd, 0);
  } else if(CCScalarField<int> *sfi =
	    dynamic_cast<CCScalarField<int> *> (_tex)) {
    cerr<<"Type = CCScalarField<int>\n";
    isCC = true;
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, sfi, 0);
  } else if(CCScalarField<long> *sfl =
	    dynamic_cast<CCScalarField<long> *> (_tex)) {
    cerr<<"Type = CCScalarField<long>\n";
    isCC = true;
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, sfl, 0);
    } */else {
    cerr<<"Error: cast didn't work!\n";
  }
#endif
  ASSERT(bontree != 0x0);
  //  AuditAllocator(default_allocator);
}

bool
GLTexture3D::SetBrickSize(int bsize)
{
  xmax = ymax = zmax = bsize;
  /*  X = _tex->nx;
  Y = _tex->ny;
  Z = _tex->nz;*/
  
  if( bontree ) delete bontree;
  computeTreeDepth();
//   bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, tex, 0);
  BuildTexture();
  return true;
}

void
GLTexture3D::SetField( void /*ScalarFieldRGBase*/ *tex )
{
  this->_tex = tex;

  /*  X = tex->nx;
  Y = tex->ny;
  Z = tex->nz;*/

  int size = std::max(X,Y);
  size = std::max(size,Z);

  /*  tex->get_bounds( minP, maxP );
      tex->get_minmax( _min, _max );*/
  xmax = ymax = zmax = 128;


// #ifdef SCI_OPENGL
//   int i;
//   int sizes[6] = { 256, 128, 64, 32, 16, 8};
//   for(i = 0; i < 5; i++){
//     if( size > sizes[i+1] ){
//       if(SetMaxBrickSize( sizes[i] ) ){
// 	break;
//       }
//     }
//   }
// #endif
   SetBounds();
   computeTreeDepth(); 
//   bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, tex, 0);
  BuildTexture();
} 
 
void GLTexture3D::computeTreeDepth()
{
  int xdepth = 0, ydepth = 0, zdepth = 0;
  int x = xmax, y = ymax, z = zmax;

  while(x < X){
    x = (x * 2) - 1;
    xdepth++;
  }
  while(y < Y){
    y = (y * 2) - 1;
    ydepth++;
  }
  while(z < Z){
    z = (z * 2) - 1;
    zdepth++;
  }

  levels = (( xdepth > ydepth)? ((xdepth > zdepth)? xdepth:
				    ((ydepth > zdepth)? ydepth:zdepth)):
	       ( ydepth > zdepth)? ydepth : zdepth);
}


bool GLTexture3D::SetMaxBrickSize(int maxBrick)
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
     xmax = ymax = zmax = maxBrick; 
     return true;
   } else {
     return false;
   }
}

template <class T>
Octree<Brick*>*
GLTexture3D::buildBonTree(Point min, Point max,
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

  if (xoff > X || yoff > Y || zoff> Z){
    node = 0;
    //return node;
  }

  Brick* brick;
  Array3<unsigned char> *brickData;
  // Check to make sure that we can accommodate the requested texture
  GLint xtex =0 , ytex = 0 , ztex = 0;

  if ( xsize <= xmax ) xtex = 1;
  if ( ysize <= ymax ) ytex = 1;
  if ( zsize <= zmax ) ztex = 1;

  brickData = scinew Array3<unsigned char>();
  int padx = 0,pady = 0,padz = 0;

  if( xtex && ytex && ztex) { // we can accommodate
    int newx = xsize, newy = ysize, newz = zsize;
    if (xsize < xmax){
      padx = xmax - xsize;
      newx = xmax;
    }
    if (ysize < ymax){
      pady = ymax - ysize;
      newy = ymax;
    }
    if (zsize < zmax){
      padz = zmax - zsize;
      newz = zmax;
    }

    makeBrickData(newx,newy,newz,xsize,ysize,zsize, xoff,yoff,zoff,
		  tex, brickData);

    brick = scinew Brick(min, max, padx,  pady, padz, level, brickData);

    node = scinew Octree<Brick*>(brick, Octree<Brick *>::LEAF,
				    parent );
  } else { // we must subdivide

    makeLowResBrickData(xmax, ymax, zmax, xsize, ysize, zsize,
			xoff, yoff, zoff, level, padx, pady, padz,
			tex, brickData);

    brick = scinew Brick(min, max, padx, pady, padz, level, brickData);

    node = scinew Octree<Brick*>(brick, Octree<Brick *>::PARENT,
				    parent);

    int sx = xmax, sy = ymax, sz = zmax, tmp;
    tmp = xmax;
    while( tmp < xsize){
      sx = tmp;
      tmp = tmp*2 -1;
    }
    tmp = ymax;
    while( tmp < ysize){
      sy = tmp;
      tmp = tmp*2 -1;
    }
    tmp = zmax;
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
    if( Z2 == Y2 && Y2 == X2 ){mid = min + Vector(dx* (sx-1), dy* (sy-1),
						  dz* (sz-1));
      for(int i = 0; i < 8; i++){
	BuildChild(i, min, mid, max, xoff, yoff, zoff,
		    xsize, ysize, zsize, sx, sy, sz,level,tex, node);
      }
    } else if( Z2 > Y2 && Z2 > X2 ) {
      mid = min + Vector(diag.x(),
			 diag.y(),
			 dz*(sz-1));
      
      BuildChild(0, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, ysize, sz, level, tex, node);
      BuildChild(1, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, ysize, sz, level, tex, node);
    } else  if( Y2 > Z2 && Y2 > X2 ) {
      mid = min + Vector(diag.x(),
			 dy*(sy - 1),
			 diag.z());
      BuildChild(0, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, zsize, level, tex, node);
      BuildChild(2, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, zsize, level, tex, node);
    } else  if( X2 > Z2 && X2 > Y2 ) {
      mid = min + Vector(dx*(sx-1),
			 diag.y(),
			 diag.z());
      BuildChild(0, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, zsize, level, tex, node);
      BuildChild(4,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, zsize, level, tex, node);
    } else if( Z2 == Y2 ){
      mid = min + Vector(diag.x(),
			 dy * (sy - 1),
			 dz* (sz - 1));
      BuildChild(0,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level, tex, node);
      BuildChild(1,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level, tex, node);
      BuildChild(2,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level, tex, node);
      BuildChild(3,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level, tex, node);
    } else if( X2 == Y2 ){
      mid = min + Vector(dx*(sx - 1), dy*(sy-1),
			 diag.z());
      BuildChild(0,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level, tex, node);
      BuildChild(2,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level, tex, node);
      BuildChild(4,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level, tex, node);
      BuildChild(6,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level, tex, node);
    } else if( Z2 == X2 ){
      mid = min + Vector(dx*(sx-1),
			 diag.y(),
			 dz*(sz-1));
      BuildChild(0,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level, tex, node);
      BuildChild(1,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level, tex, node);
      BuildChild(4,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level, tex, node);
      BuildChild(5,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level, tex, node);
    }
  }
  return node;
}

template <class T>
void GLTexture3D::BuildChild(int i, Point min, Point mid, Point max,
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
    node->SetChild(0, buildBonTree(pmin, pmax, xoff, yoff, zoff,
				   X2, Y2, Z2, level, tex, node));
    break;
  case 1:
    pmin = min;
    pmax = mid;
    pmin.z(mid.z());
    pmax.z(max.z());
    node->SetChild(1, buildBonTree(pmin, pmax,
				   xoff, yoff, zoff + Z2 -1,
				   X2, Y2, zsize-Z2+1, level, tex, node));
    break;
  case 2:
    pmin = min;
    pmax = mid;
    pmin.y(mid.y());
    pmax.y(max.y());
    node->SetChild(2, buildBonTree(pmin, pmax,
				   xoff, yoff + Y2 - 1, zoff,
				   X2, ysize - Y2 + 1, Z2, level, tex, node));
    break;
  case 3:
    pmin = mid;
    pmax = max;
    pmin.x(min.x());
    pmax.x(mid.x());
    node->SetChild(3, buildBonTree(pmin, pmax,
				   xoff, yoff + Y2 - 1 , zoff + Z2 - 1,
				   X2, ysize - Y2 + 1, zsize - Z2 + 1, level, tex, node));
    break;
  case 4:
    pmin = min;
    pmax = mid;
    pmin.x(mid.x());
    pmax.x(max.x());
    node->SetChild(4, buildBonTree(pmin, pmax,
				   xoff + X2 - 1, yoff, zoff,
				   xsize - X2 + 1, Y2, Z2, level, tex, node));
    break;
  case 5:
    pmin = mid;
    pmax = max;
    pmin.y(min.y());
    pmax.y(mid.y());
    node->SetChild(5, buildBonTree(pmin, pmax,
				   xoff + X2 - 1, yoff, zoff +  Z2 - 1,
				   xsize - X2 + 1, Y2, zsize - Z2 + 1, level, tex, node));
    break;
  case 6:
    pmin = mid;
    pmax = max;
    pmin.z(min.z());
    pmax.z(mid.z());
    node->SetChild(6, buildBonTree(pmin, pmax,
				   xoff + X2 - 1, yoff + Y2 - 1, zoff,
				   xsize - X2 + 1, ysize - Y2 + 1, Z2, level, tex, node));
    break;
  case 7:
   pmin = mid;
   pmax = max;
   node->SetChild(7, buildBonTree(pmin, pmax,  xoff + X2 - 1,
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
  double v = (val - _min)*255/(_max - _min);
  if ( v < 0 ) return 0;
  else if (v > 255) return 255;
  else return v;
}

template <class T>
void GLTexture3D::makeBrickData(int newx, int newy, int newz,
				int xsize, int ysize, int zsize,
				int xoff, int yoff, int zoff, T *tex,
				Array3<unsigned char>*& bd)
{
  int i,j,k,ii,jj,kk;

  bd->newsize( newz, newy, newx);
  for(kk = 0, k = zoff; kk < zsize; kk++, k++)
    for(jj = 0, j = yoff; jj < ysize; jj++, j++)
      for(ii = 0, i = xoff; ii < xsize; ii++, i++){
	(*bd)(kk,jj,ii) = SETVAL( tex->grid(i,j,k) );
  }

}

template <class T>						
void GLTexture3D::makeLowResBrickData(int xmax, int ymax, int zmax,
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
	  k00 = Interpolate(SETVAL( tex->grid(i,j,k) ),
			    SETVAL( tex->grid(i,j,k1)),dk);
	  k01 = Interpolate(SETVAL( tex->grid(i1,j,k)),
			    SETVAL( tex->grid(i1,j,k1)),dk);
	  k10 = Interpolate(SETVAL( tex->grid(i,j1,k)),
			    SETVAL( tex->grid(i,j,k1)),dk);
	  k11 = Interpolate(SETVAL( tex->grid(i1,j1,k)),
			    SETVAL( tex->grid(i1,j1,k1)),dk);
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
      dx = pow(2.0, levels - level);
      if( xmax * dx > xsize){
	padx = (xmax*dx - xsize)/dx;
      }
    }
    if( ymax > ysize ) {
      dy = 1; pady = (ymax - ysize);
    } else {
      dy = pow(2.0, levels - level);
      if( ymax * dy > ysize){
	pady = (ymax*dy - ysize)/dy;
      }
    }
    if( zmax > zsize ) {
      dz = 1; padz = (zmax - zsize);
    } else {
      dz = pow(2.0, levels - level);
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
	    (*bd)(kk,jj,ii) = SETVAL( tex->grid(i,j,k) );
	  }
	}
      }
  }    
}


} // End namespace SCIRun
