#include "GLTexture3D.h"
#include "Brick.h"
#include "VolumeUtils.h"
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Persistent/Persistent.h>
#include <SCICore/Containers/String.h>


#include <GL/gl.h>
#include <iostream>
#include <string>





namespace Kurt {
namespace Datatypes {

using SCICore::Datatypes::ScalarField;
using SCICore::Datatypes::Persistent;
using SCICore::Containers::clString;
using std::cerr;
using std::endl;
using std::string;

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

PersistentTypeID GLTexture3D::type_id("ScalarFieldRGuchar", "ScalarField"
, maker);
#define GLTexture3D_VERSION 3
void GLTexture3D::io(Piostream&)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;
    using SCICore::Geometry::Pio;
    NOT_FINISHED("GLTexture3D::io(Piostream&)");
}
GLTexture3D::GLTexture3D() :
  tex(0), X(0), Y(0),
  Z(0), xmax(0), ymax(0), zmax(0)
{
}

GLTexture3D::GLTexture3D(ScalarFieldRGuchar *tex ) :
  tex(tex), X(tex->grid.dim3()), Y(tex->grid.dim2()),
  Z(tex->grid.dim1()),  xmax(64), ymax(64), zmax(64)
{
  int size = std::max(X,Y);
  size = std::max(size,Z);
  tex->get_bounds( minP, maxP );

  Vector diag = maxP - minP;

  dx = diag.x()/(X-1);
  dy = diag.y()/(Y-1);
  dz = diag.z()/(Z-1); 

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
  
  computeTreeDepth(); 
  bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,0);

}

bool
GLTexture3D::SetBrickSize(int bsize)
{
  xmax = ymax = zmax = bsize;
  X = tex->grid.dim3();
  Y = tex->grid.dim2();
  Z = tex->grid.dim1();
  
  if( bontree ) delete bontree;
  computeTreeDepth();
  bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,0);
  return true;
}

void
GLTexture3D::SetField( ScalarFieldRGuchar *tex )
{
  this->tex = tex;
  X = tex->grid.dim3();
  Y = tex->grid.dim2();
  Z = tex->grid.dim1();

  int size = std::max(X,Y);
  size = std::max(size,Z);

  tex->get_bounds( minP, maxP );
  xmax = ymax = zmax = 64;

  Vector diag = maxP - minP;

  dx = diag.x()/(X-1);
  dy = diag.y()/(Y-1);
  dz = diag.z()/(Z-1); 

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
  
  computeTreeDepth(); 
  bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,0);
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
   glTexImage3D(GL_PROXY_TEXTURE_3D, 0, GL_INTENSITY8_EXT, x, y, z, 0, 
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
   std::cerr<<xtex<<" "<<ytex<<" "<<ztex<<std::endl;
   if( xtex && ytex) { // we can accommodate 
     xmax = ymax = zmax = maxBrick; 
     return true;
   } else {
     return false;
   }
}

Octree<Brick*>*
GLTexture3D::buildBonTree(Point min, Point max,
            int xoff, int yoff, int zoff,
            int xsize, int ysize, int zsize,
	    int level, Octree<Brick*>* parent)
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

  brickData = new Array3<unsigned char>();
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

    makeBrickData(newx,newy,newz,xsize,ysize,zsize, xoff,yoff,zoff, brickData);

    brick = new Brick(min, max, padx,  pady, padz, level, brickData);

    node = new Octree<Brick*>(brick, Octree<Brick *>::LEAF,
				    parent );
  } else { // we must subdivide

    makeLowResBrickData(xmax, ymax, zmax, xsize, ysize, zsize,
			xoff, yoff, zoff, level, padx, pady, padz, brickData);

    brick = new Brick(min, max, padx, pady, padz, level, brickData);

    node = new Octree<Brick*>(brick, Octree<Brick *>::PARENT,
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
		    xsize, ysize, zsize, sx, sy, sz,level,node);
      }
    } else if( Z2 > Y2 && Z2 > X2 ) {
      mid = min + Vector(diag.x(),
			 diag.y(),
			 dz*(sz-1));
      
      BuildChild(0, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, ysize, sz, level, node);
      BuildChild(1, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, ysize, sz, level, node);
    } else  if( Y2 > Z2 && Y2 > X2 ) {
      mid = min + Vector(diag.x(),
			 dy*(sy - 1),
			 diag.z());
      BuildChild(0, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, zsize, level, node);
      BuildChild(2, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, zsize, level, node);
    } else  if( X2 > Z2 && X2 > Y2 ) {
      mid = min + Vector(dx*(sx-1),
			 diag.y(),
			 diag.z());
      BuildChild(0, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, zsize, level, node);
      BuildChild(4,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, zsize, level, node);
    } else if( Z2 == Y2 ){
      mid = min + Vector(diag.x(),
			 dy * (sy - 1),
			 dz* (sz - 1));
      BuildChild(0,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level, node);
      BuildChild(1,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level, node);
      BuildChild(2,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level, node);
      BuildChild(3,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level, node);
    } else if( X2 == Y2 ){
      mid = min + Vector(dx*(sx - 1), dy*(sy-1),
			 diag.z());
      BuildChild(0,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level, node);
      BuildChild(2,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level, node);
      BuildChild(4,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level, node);
      BuildChild(6,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level, node);
    } else if( Z2 == X2 ){
      mid = min + Vector(dx*(sx-1),
			 diag.y(),
			 dz*(sz-1));
      BuildChild(0,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level, node);
      BuildChild(1,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level, node);
      BuildChild(4,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level, node);
      BuildChild(5,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level, node);
    }
  }
  return node;
}

void GLTexture3D::BuildChild(int i, Point min, Point mid, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level,  Octree<Brick*>* node)
{
  Point pmin, pmax;

  switch( i ) {
  case 0:
    pmin = min;
    pmax = mid;
    node->SetChild(0, buildBonTree(pmin, pmax, xoff, yoff, zoff,
				   X2, Y2, Z2, level, node));
    break;
  case 1:
    pmin = min;
    pmax = mid;
    pmin.z(mid.z());
    pmax.z(max.z());
    node->SetChild(1, buildBonTree(pmin, pmax,
				   xoff, yoff, zoff + Z2 -1,
				   X2, Y2, zsize-Z2+1, level, node));
    break;
  case 2:
    pmin = min;
    pmax = mid;
    pmin.y(mid.y());
    pmax.y(max.y());
    node->SetChild(2, buildBonTree(pmin, pmax,
				   xoff, yoff + Y2 - 1, zoff,
				   X2, ysize - Y2 + 1, Z2, level, node));
    break;
  case 3:
    pmin = mid;
    pmax = max;
    pmin.x(min.x());
    pmax.x(mid.x());
    node->SetChild(3, buildBonTree(pmin, pmax,
				   xoff, yoff + Y2 - 1 , zoff + Z2 - 1,
				   X2, ysize - Y2 + 1, zsize - Z2 + 1, level, node));
    break;
  case 4:
    pmin = min;
    pmax = mid;
    pmin.x(mid.x());
    pmax.x(max.x());
    node->SetChild(4, buildBonTree(pmin, pmax,
				   xoff + X2 - 1, yoff, zoff,
				   xsize - X2 + 1, Y2, Z2, level, node));
    break;
  case 5:
    pmin = mid;
    pmax = max;
    pmin.y(min.y());
    pmax.y(mid.y());
    node->SetChild(5, buildBonTree(pmin, pmax,
				   xoff + X2 - 1, yoff, zoff +  Z2 - 1,
				   xsize - X2 + 1, Y2, zsize - Z2 + 1, level, node));
    break;
  case 6:
    pmin = mid;
    pmax = max;
    pmin.z(min.z());
    pmax.z(mid.z());
    node->SetChild(6, buildBonTree(pmin, pmax,
				   xoff + X2 - 1, yoff + Y2 - 1, zoff,
				   xsize - X2 + 1, ysize - Y2 + 1, Z2, level, node));
    break;
  case 7:
   pmin = mid;
   pmax = max;
   node->SetChild(7, buildBonTree(pmin, pmax,  xoff + X2 - 1,
				  yoff + Y2 - 1, zoff +  Z2 - 1,
				  xsize - X2 + 1, ysize - Y2 + 1,
				  zsize - Z2 + 1, level, node));
   break;
  default:
    break;
  }
}



void GLTexture3D::makeBrickData(int newx, int newy, int newz,
			       int xsize, int ysize, int zsize,
			       int xoff, int yoff, int zoff,
			       Array3<unsigned char>*& bd)
{
  int i,j,k,ii,jj,kk;

  bd->newsize( newz, newy, newx);
  for(kk = 0, k = zoff; kk < zsize; kk++, k++)
    for(jj = 0, j = yoff; jj < ysize; jj++, j++)
      for(ii = 0, i = xoff; ii < xsize; ii++, i++){
	(*bd)(kk,jj,ii) = tex->grid(k,j,i);
  }

}
						
void GLTexture3D::makeLowResBrickData(int xmax, int ymax, int zmax,
				     int xsize, int ysize, int zsize,
				     int xoff, int yoff, int zoff,
				     int level, int& padx, int& pady,
				     int& padz, Array3<unsigned char>*& bd)
{
  using SCICore::Math::Interpolate;

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
	  k00 = Interpolate(tex->grid(k,j,i),tex->grid(k,j,i1),dk);
	  k01 = Interpolate(tex->grid(k1,j,i),tex->grid(k1,j,i1),dk);
	  k10 = Interpolate(tex->grid(k,j1,i),tex->grid(k,j,i1),dk);
	  k11 = Interpolate(tex->grid(k1,j1,i),tex->grid(k1,j1,i1),dk);
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
	    (*bd)(kk,jj,ii) = tex->grid(k,j,i);
	  }
	}
      }
  }    
}


} // end namespace Datatypes
} // end namespace Kurt
