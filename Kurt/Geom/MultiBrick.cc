
#include <SCICore/Geom/GeomOpenGL.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Geometry/Ray.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Math/MiscMath.h>
#include "MultiBrick.h"
#include "stdlib.h"
#include "VolumeUtils.h"
#include <iostream>
#include <string>
namespace SCICore {
namespace GeomSpace {

using namespace SCICore::Geometry;
using namespace Kurt::GeomSpace; 
using std::cerr;
using std::endl;
 using std::string;


int MultiBrick::traversalTable[27][8] = { {7,3,5,6,1,2,4,0},
					  {6,7,2,3,4,5,0,1},
					  {6,2,4,7,0,3,5,1},
					  {5,7,1,3,4,6,0,2},
					  {4,5,6,7,0,1,2,3},
					  {4,6,0,2,5,7,1,3},
					  {5,1,4,7,0,3,6,2},
					  {4,5,0,1,6,7,2,3},
					  {4,0,5,6,1,2,7,3},
					  {3,7,1,2,5,6,0,4},
					  {2,3,6,7,0,1,4,5},
					  {2,6,0,3,4,7,1,5},
					  {1,3,5,7,0,2,4,6},
					  {0,1,2,3,4,5,6,7},
					  {0,2,4,6,1,3,5,7},
					  {1,5,0,3,4,7,2,6},
					  {0,1,4,5,2,3,6,7},
					  {0,4,1,2,5,6,3,7},
					  {3,1,2,7,0,5,6,4},
					  {2,3,0,1,6,7,4,5},
					  {2,0,3,6,1,4,7,5},
					  {1,3,0,2,5,7,4,6},
					  {0,1,2,3,4,5,6,7},
					  {0,2,1,3,4,6,5,7},
					  {1,0,3,5,2,4,7,6},
					  {0,1,2,3,4,5,6,7},
					  {0,1,2,4,3,5,6,7}};



  
MultiBrick::MultiBrick(int id, int slices, double alpha,
		       int maxdim, Point min, Point max,
		       int mode, bool debug,
		       int X, int Y, int Z,
		       const ScalarFieldRGuchar* tex,
		       const GLvoid* cmap) :
  GeomObj(id), alpha(alpha),  slices(slices),
  tex( tex ), cmap(cmap), min(min), max(max), debug(debug),
  X(X), Y(Y), Z(Z), mode(mode), drawWireFrame( false ),
  drawLevel(0), reload( (unsigned char*)&tex->grid(0,0,0)),
  xmax(maxdim),ymax(maxdim),zmax(maxdim), treeDepth(0), nodeId(0)
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

  //cerr<<min<<", "<<max<<", "<<X<<", "<<Y<<", "<<Z<<endl;
Vector diag = max - min;
dx = diag.x()/(X-1);
dy = diag.y()/(Y-1);
dz = diag.z()/(Z-1); 
//cerr<<"(dx, dy, dz) = ("<<dx<<", "<<dy<<", "<<dz<<" )\n";
#ifdef SCI_OPENGL
//octree = buildOctree(min, max, 0, 0, 0, X, Y, Z, 0);
  computeTreeDepth(); 
  nodeId = 0; 
  octree = buildBonTree(min, max, 0, 0, 0, X, Y, Z, 0,0);
#endif

}
 
void
MultiBrick::SetVol( const ScalarFieldRGuchar *tex )
{
  this->tex = tex;
  X = tex->nx;
  Y = tex->ny;
  Z = tex->nz;
  
  Vector diag = max - min;
  dx = diag.x()/(X-1);
  dy = diag.y()/(Y-1);
  dz = diag.z()/(Z-1);
#ifdef SCI_OPENGL
//octree = buildOctree(min, max, 0, 0, 0, X, Y, Z, 0);
  computeTreeDepth(); 
  nodeId = 0; 
  octree = buildBonTree(min, max, 0, 0, 0, X, Y, Z, 0,0);
#endif

}

MultiBrick::~MultiBrick()
{
#ifdef SCI_OPENGL
  //glDeleteTextures(1, &texName );
#endif
}

void MultiBrick::computeTreeDepth()
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

  treeDepth = (( xdepth > ydepth)? ((xdepth > zdepth)? xdepth:
				    ((ydepth > zdepth)? ydepth:zdepth)):
	       ( ydepth > zdepth)? ydepth : zdepth);
}
  

void MultiBrick::SetMaxBrickSize(int x,int y,int z)
{
/*   GLint xtex, ytex, ztex; */
/*   glTexImage3DEXT(GL_PROXY_TEXTURE_3D_EXT, 0, 4, x, y, z, 0, */
/*                     GL_RGBA, GL_UNSIGNED_BYTE, tex); */
/*   glGetTexLevelParameteriv( GL_PROXY_TEXTURE_3D_EXT, 0, */
/*                             GL_TEXTURE_WIDTH, &xtex); */
/*   glGetTexLevelParameteriv( GL_PROXY_TEXTURE_3D_EXT, 0, */
/*                             GL_TEXTURE_HEIGHT, &ytex); */
/*   glGetTexLevelParameteriv( GL_PROXY_TEXTURE_3D_EXT, 0, */
/*                             GL_TEXTURE_DEPTH_EXT, &ztex); */

/*   if( xtex && ytex && ztex) { // we can accommodate */
    xmax = x; ymax = y; zmax = z;
/*   } */
}

VolumeOctree<Brick*>*
MultiBrick::buildBonTree(Point min, Point max,
            int xoff, int yoff, int zoff,
            int xsize, int ysize, int zsize,
	    int level, int id)
{
  //cerr<<"level "<< level<<endl;
  //    cerr<<min<<", "<<max<<", "<<xsize<<", "<<ysize<<", "<<zsize<<endl;

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

  VolumeOctree<Brick *> *node;

  if (xoff > X || yoff > Y || zoff> Z){
    node = 0;
    //return node;
  }

  Brick* brick;
  Array3<unsigned char> *brickData;
  // Check to make sure that we can accommodate the requested texture
  GLint xtex =0 , ytex = 0 , ztex = 0;
/*   glTexImage3DEXT(GL_PROXY_TEXTURE_3D_EXT, 0, 4, xsize, ysize, zsize, 0, */
/*                     GL_RGBA, GL_UNSIGNED_BYTE, tex); */
/*   glGetTexLevelParameteriv( GL_PROXY_TEXTURE_3D_EXT, 0, */
/*                             GL_TEXTURE_WIDTH, &xtex); */
/*   glGetTexLevelParameteriv( GL_PROXY_TEXTURE_3D_EXT, 0, */
/*                             GL_TEXTURE_HEIGHT, &ytex); */
/*   glGetTexLevelParameteriv( GL_PROXY_TEXTURE_3D_EXT, 0, */
/*                             GL_TEXTURE_DEPTH_EXT, &ztex); */
  if ( xsize <= xmax ) xtex = 1;
  if ( ysize <= ymax ) ytex = 1;
  if ( zsize <= zmax ) ztex = 1;

  brickData = new Array3<unsigned char>();
    int padx = 0,pady = 0,padz = 0;

  if( xtex && ytex && ztex) { // we can accommodate
/*     if (!isPowerOf2( xsize )){ */
/*       int newx =  nextPowerOf2( xsize ); */
/*       padx = newx - xsize; */
/*     } */
/*     if (!isPowerOf2( ysize )){ */
/*       int newy =  nextPowerOf2( ysize ); */
/*       pady = newy - ysize; */
/*     } */
/*     if (!isPowerOf2( zsize )){ */
/*       int newz =  nextPowerOf2( zsize ); */
/*       padz = newz - zsize; */
/*     } */
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
    brick = new Brick(min, max, padx, pady, padz, level,
		      1.0/pow(2.0,level),true, debug,  brickData);

    node = new VolumeOctree<Brick*>(min, max, brick, id,
                                     VolumeOctree<Brick *>::LEAF );
  } else { // we must subdivide

    makeLowResBrickData(xmax, ymax, zmax, xsize, ysize, zsize,
		    xoff, yoff, zoff, level, padx, pady, padz,brickData);

    brick = new Brick(min, max, padx, pady, padz, level,
		      1.0/pow(2.0,level),true, debug,  brickData);
    node = new VolumeOctree<Brick*>(min, max, brick, id,
				    VolumeOctree<Brick *>::PARENT );

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

/*     int nbx, nby, nbz; */
/*     nbx = largestPowerOf2( xsize/xmax ); */
/*      nby = largestPowerOf2(ysize/ymax); */
/*      nbz = largestPowerOf2(zsize/zmax); */
/*      int sx, sy, sz; */
/*      sx = xmax + (xmax-1)*(nbx-1); */
/*      sy = ymax + (ymax-1)*(nby-1); */
/*      sz = zmax + (zmax-1)*(nbz-1); */

    int X2, Y2, Z2;
    X2 = largestPowerOf2( xsize -1);
    Y2 = largestPowerOf2( ysize -1);
    Z2 = largestPowerOf2( zsize -1);


    Vector diag = max - min;
    Point mid;
    if( Z2 == Y2 && Y2 == X2 ){mid = min + Vector(dx* (sx-1), dy* (sy-1),
						  dz* (sz-1));
      for(int i = 0; i < 8; i++){
	BuildChild(i, (id*8)+(i+1), min, mid, max, xoff, yoff, zoff,
		    xsize, ysize, zsize, sx, sy, sz,level,node);
      }
    } else if( Z2 > Y2 && Z2 > X2 ) {
      mid = min + Vector(diag.x(),
			 diag.y(),
			 dz*(sz-1));
		    //			 diag.z() * (sz-1.0)/(zsize-1.0));
      
      BuildChild(0, (id*8)+1, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, ysize, sz, level, node);
      BuildChild(1, (id*8)+2, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, ysize, sz, level, node);
    } else  if( Y2 > Z2 && Y2 > X2 ) {
      mid = min + Vector(diag.x(),
			 //diag.y() * (sy-1.0)/(ysize-1.0),
			 dy*(sy - 1),
			 diag.z());
      BuildChild(0, (id*8)+1, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, zsize, level, node);
      BuildChild(2, (id*8)+3,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, zsize, level, node);
    } else  if( X2 > Z2 && X2 > Y2 ) {
      mid = min + Vector(//diag.x() * (sx-1.0)/(xsize-1.0),
			 dx*(sx-1),
			 diag.y(),
			 diag.z());
      BuildChild(0, (id*8)+1, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, zsize, level, node);
      BuildChild(4, (id*8)+5,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, zsize, level, node);
    } else if( Z2 == Y2 ){
      mid = min + Vector(diag.x(),
			 //diag.y() * (sy-1.0)/(ysize-1.0),
			 dy * (sy - 1),
			 dz* (sz - 1));
      //diag.z() * (sz-1.0)/(zsize-1.0));
      BuildChild(0, (id*8)+1,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level, node);
      BuildChild(1, (id*8)+2,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level, node);
      BuildChild(2,(id*8)+3,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level, node);
      BuildChild(3,(id*8)+4,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level, node);
    } else if( X2 == Y2 ){
      mid = min + Vector(dx*(sx - 1), dy*(sy-1),
			 //diag.x() * (sx-1.0)/(xsize-1.0),
			 //diag.y() * (sy-1.0)/(ysize-1.0),
			 diag.z());
      BuildChild(0,(id*8)+1,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level, node);
      BuildChild(2,(id*8)+3,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level, node);
      BuildChild(4,(id*8)+5,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level, node);
      BuildChild(6,(id*8)+7,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level, node);
    } else if( Z2 == X2 ){
      mid = min + Vector(dx*(sx-1),
			 //diag.x() * (sx-1.0)/(xsize-1.0),
			 diag.y(),
			 dz*(sz-1));
			 //diag.z() * (sz-1.0)/(zsize-1.0));
      BuildChild(0,(id*8)+1,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level, node);
      BuildChild(1,(id*8)+2,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level, node);
      BuildChild(4,(id*8)+5,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level, node);
      BuildChild(5,(id*8)+6,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level, node);
    }
  }
  return node;
}

void MultiBrick::BuildChild(int i, int id,  Point min, Point mid, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level,  VolumeOctree<Brick*>* node)
{
  Point pmin, pmax;

  switch( i ) {
  case 0:
    pmin = min;
    pmax = mid;
    node->SetChild(0, buildBonTree(pmin, pmax, xoff, yoff, zoff,
				   X2, Y2, Z2, level, id));
    break;
  case 1:
    pmin = min;
    pmax = mid;
    pmin.z(mid.z());
    pmax.z(max.z());
    node->SetChild(1, buildBonTree(pmin, pmax,
				   xoff, yoff, zoff + Z2 -1,
				   X2, Y2, zsize-Z2+1, level, id));
    break;
  case 2:
    pmin = min;
    pmax = mid;
    pmin.y(mid.y());
    pmax.y(max.y());
    node->SetChild(2, buildBonTree(pmin, pmax,
				   xoff, yoff + Y2 - 1, zoff,
				   X2, ysize - Y2 + 1, Z2, level, id));
    break;
  case 3:
    pmin = mid;
    pmax = max;
    pmin.x(min.x());
    pmax.x(mid.x());
    node->SetChild(3, buildBonTree(pmin, pmax,
				   xoff, yoff + Y2 - 1 , zoff + Z2 - 1,
				   X2, ysize - Y2 + 1, zsize - Z2 + 1, level, id));
    break;
  case 4:
    pmin = min;
    pmax = mid;
    pmin.x(mid.x());
    pmax.x(max.x());
    node->SetChild(4, buildBonTree(pmin, pmax,
				   xoff + X2 - 1, yoff, zoff,
				   xsize - X2 + 1, Y2, Z2, level, id));
    break;
  case 5:
    pmin = mid;
    pmax = max;
    pmin.y(min.y());
    pmax.y(mid.y());
    node->SetChild(5, buildBonTree(pmin, pmax,
				   xoff + X2 - 1, yoff, zoff +  Z2 - 1,
				   xsize - X2 + 1, Y2, zsize - Z2 + 1, level, id));
    break;
  case 6:
    pmin = mid;
    pmax = max;
    pmin.z(min.z());
    pmax.z(mid.z());
    node->SetChild(6, buildBonTree(pmin, pmax,
				   xoff + X2 - 1, yoff + Y2 - 1, zoff,
				   xsize - X2 + 1, ysize - Y2 + 1, Z2, level, id));
    break;
  case 7:
   pmin = mid;
   pmax = max;
   node->SetChild(7, buildBonTree(pmin, pmax,  xoff + X2 - 1,
				  yoff + Y2 - 1, zoff +  Z2 - 1,
				  xsize - X2 + 1, ysize - Y2 + 1,
				  zsize - Z2 + 1, level, id));
   break;
  default:
    break;
  }
}



void MultiBrick::makeBrickData(int newx, int newy, int newz,
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
						
void MultiBrick::makeLowResBrickData(int xmax, int ymax, int zmax,
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
    dx = (double)(xsize-1)/(xmax-1);
    dy = (double)(ysize-1)/(ymax-1);
    dz = (double)(zsize-1)/(zmax-1);

    //  cerr<<"xsize x ysize x zsize ="<<xsize<<"x"<<ysize<<"x"<<zsize<<endl;
    //  cerr<<"xmax x ymax x zmax ="<<xmax<<"x"<<ymax<<"x"<<zmax<<endl;  
    //  cerr<<"xoff x yoff x zoff ="<<xoff<<"x"<<yoff<<"x"<<zoff<<endl;
    int k1,j1,i1;
    double dk,dj,di, k00,k01,k10,k11,j00,j01;

    for(kk = 0, k = zoff; kk < zmax; kk++, k+=dz){
      k1 = ((int)k + 1 >= zoff+zsize)?k:(int)k + 1 ;
      if(k1 == (int)k ) { dk = 0; } else {dk = k1 - k;}
      for(jj = 0,j = yoff; jj < ymax; jj++, j+=dy){
	j1 = ((int)j + 1 >= zoff+zsize)?j:(int)j + 1 ;
	if(j1 == (int)j) {dj = 0;} else { dj = j1 - j;} 
	for(ii = 0, i = xoff; ii < xmax; ii++, i+=dx){
	  i1 = ((int)i + 1 >= zoff+zsize)?i:(int)i + 1 ;
	  if( i1 == (int)i){ di = 0;} else {di = i1 - i;}
	  k00 = Interpolate(tex->grid(k,j,i),tex->grid(k1,j,i),dk);
	  k01 = Interpolate(tex->grid(k,j,i1),tex->grid(k1,j,i1),dk);
	  k10 = Interpolate(tex->grid(k,j1,i1),tex->grid(k1,j1,i),dk);
	  k11 = Interpolate(tex->grid(k,j1,i1),tex->grid(k1,j,i1),dk);
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
      dx = pow(2.0, treeDepth - level);
      if( xmax * dx > xsize){
	padx = (xmax*dx - xsize)/dx;
      }
    }
    if( ymax > ysize ) {
      dy = 1; pady = (ymax - ysize);
    } else {
      dy = pow(2.0, treeDepth - level);
      if( ymax * dy > ysize){
	pady = (ymax*dy - ysize)/dy;
      }
    }
    if( zmax > zsize ) {
      dz = 1; padz = (zmax - zsize);
    } else {
      dz = pow(2.0, treeDepth - level);
      if( zmax * dz > zsize){
	padz = (zmax*dz - zsize)/dz;
      }
    }
    if(debug){
      cerr<<"xmax = "<< xmax * dx<<", xsize = "<<xsize<<endl;
      cerr<<"ymax = "<< ymax * dy<<", ysize = "<<ysize<<endl;
      cerr<<"zmax = "<< zmax * dz<<", zsize = "<<zsize<<endl;
      cerr<<"dx, dy, dz = "<< dx<<", "<<dy<<", "<<dz<<endl;
      cerr<<"padx, pady, padz = "<< padx<<", "<<pady<<", "<<padz<<endl;
    }
/*     dx = pow(2.0, treeDepth - level); */
/*     dy = pow(2.0, treeDepth - level); */
/*     dz = pow(2.0, treeDepth - level); */
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
}





#ifdef SCI_OPENGL
void 
MultiBrick::draw(DrawInfoOpenGL* di, Material* mat, double time)
{
  if( !pre_draw(di, mat, 0) ) return;

  if ( di->get_drawtype() == DrawInfoOpenGL::WireFrame ) {
    drawWireFrame = true;
  } else {
    drawWireFrame = false;
  }

  drawSlices();

}  
#endif

MultiBrick::MultiBrick(const MultiBrick& copy)
: GeomObj(copy.id), slices(copy.slices)
{
}


GeomObj* MultiBrick::clone()
{
    return new MultiBrick(*this);
}

void MultiBrick::get_bounds(BBox& bb)
{
  bb.extend( min );
  bb.extend( max );
}
  


void 
MultiBrick::drawSlices()
{


  double mvmat[16];
  Transform mat;
  Vector view;
  Point viewPt;
  Ray viewRay;
      
  glGetDoublev( GL_MODELVIEW_MATRIX, mvmat);
  glPrintError("glGetDoublev( GL_MODELVIEW_MATRIX, mvmat)");
  /* remember that the glmatrix is stored as
       0  4  8 12
       1  5  9 13
       2  6 10 14
       3  7 11 15 */
 
  view = Vector(mvmat[12], mvmat[13], mvmat[14]);
  view.normalize();
  viewPt = Point(-mvmat[12], -mvmat[13], -mvmat[14]);
    
  /* set the translation to zero */
  mvmat[12] = mvmat[13] = mvmat[14] = 0;
  /* Because of the order of the glmatrix we are storing as a transpose.
       if there is not use of scale then the transpose is the  inverse */
  mat.set( mvmat );
    
  /* project view info into object space */
  view = mat.project( view );
  viewPt = mat.project( viewPt );
  viewRay = Ray(viewPt, view);

  // Slice the volume---use GL_TEXTURE_GEN to generate texture coords.
  glDisable(GL_DEPTH_TEST);
  glPrintError("glDisable(GL_DEPTH_TEST)");
  glEnable(GL_TEXTURE_3D_EXT);
  glPrintError("glEnable(GL_TEXTURE_3D_EXT)");
  glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE, GL_MODULATE); 
  glPrintError("glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE, GL_MODULATE)");
  //glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE, GL_REPLACE);
  if( cmap ) {
#ifdef __sgi
    //cerr << "Using Lookup!\n";
    glEnable(GL_TEXTURE_COLOR_TABLE_SGI);
    glPrintError("glEnable(GL_TEXTURE_COLOR_TABLE_SGI)");
    glColorTableSGI(GL_TEXTURE_COLOR_TABLE_SGI,
		    GL_RGBA,
		    256, // try larger sizes?
		    GL_RGBA,  // need an alpha value...
		    GL_UNSIGNED_BYTE, // try shorts...
		    cmap);
    glPrintError("glColorTableSGI");
#endif
  }
  glColor4f(1,1,1,1); // set to all white for modulation
  glPrintError("glColor4f(1,1,1,1)");
  

  glEnable(GL_BLEND);
  glPrintError("glEnable(GL_BLEND)");

  // Maximum Intensity Projections
  if( mode != 3 ) {
    if( mode == 2 ){
      glBlendEquationEXT(GL_MAX_EXT);
      glPrintError("glBlendEquationEXT(GL_MAX_EXT)");
      glBlendFunc(GL_ONE, GL_ONE);  //glBlendFunc(GL_ONE, GL_ONE);
      glPrintError("glBlendFunc(GL_ONE, GL_ONE)");
    } else if( mode == 1) {
      glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
      glPrintError("glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)");
    } else {
      //glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE, GL_REPLACE); 
      //glBlendFunc(GL_SRC_ALPHA, GL_ONE);
      glDisable(GL_BLEND);
      glEnable(GL_DEPTH_TEST);
    }
    // This combo works
    //glBlendEquationEXT(GL_FUNC_ADD_EXT);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    
    
    SliceTable st(min,max, viewRay, treeDepth, slices);
    drawBonTree( octree, viewRay, st);
    /*   drawBonTree( octree, viewRay); */
    // NOT_FINISHED("MultiBrick::drawSlices()");
  }
  if ( mode ){
    glDepthMask(GL_TRUE);
    glBlendEquationEXT(GL_FUNC_ADD_EXT);
  }
  glDisable(GL_BLEND);
#ifdef __sgi
  if( cmap )
    glDisable(GL_TEXTURE_COLOR_TABLE_SGI);
#endif
  glDisable(GL_TEXTURE_3D_EXT);
  glEnable(GL_DEPTH_TEST);  
}
void MultiBrick::drawBonTree( const VolumeOctree<Brick*>* node, 
			     const Ray&  viewRay, const SliceTable& st)
{
  int i;
  //double  ts[8];
  double tmin, tmax, dt;

  if ( node == NULL ) return;
   
  Brick* brick = (*node)(); // get the contents of the node
  ////cerr<<", level = "<<brick->getLevel()<<endl;
  if( node->getType() == VolumeOctree<Brick*>::LEAF ){
    st.getParameters( brick, tmin, tmax, dt );
    if(debug)
      cerr<<"drawing node "<< node->getId()<<" at level "<<brick->getLevel()<<endl;
    if(!mode) {
    brick->draw(viewRay, drawWireFrame, reload != 0,  widgetPoint );
  } else {
    brick->draw( viewRay, alpha, drawWireFrame, reload != 0, tmin, tmax, dt);/*     for( i = 0; i < 8; i++) */
  }
    if( reload !=0 )
      reload = 0;
/*       ts[i] = intersectParam(-viewRay.direction(), */
/* 			     brick->getCorner(i), viewRay); */
/*     sortParameters(ts,8); */
/*     brick->draw( viewRay, alpha, drawWireFrame, ts[7], ts[0], (ts[0]-ts[7])/slices ); */

  } else {
    int *traversal;
    int traversalIndex, x, y, z;
    Point min, max, mid;
    const VolumeOctree<Brick*>* child;
    child = node->child(0);
    
    mid = child->getMax();
    min = child->getMin();
    if(debug){
      cerr<<"PARENT node "<< node->getId()<<endl;
      //cerr<<"child 0: min,  max = "<<min<<", "<<mid<<endl;
    }


    if( viewRay.origin().x() < mid.x()) x = 0;
    else if( viewRay.origin().x() == mid.x()) x = 1;
    else x = 2;
    if( viewRay.origin().y() < mid.y()) y = 0;
    else if( viewRay.origin().y() == mid.y()) y = 1;
    else y = 2;
    if( viewRay.origin().z() < mid.z()) z = 0;
    else if( viewRay.origin().z() == mid.z()) z = 1;
    else z = 2;
    
    traversalIndex = 9*x + 3*y + z;
    if(debug){
      cerr<<"Traversal index = "<<traversalIndex<<endl;
    }

    traversal = traversalTable[ traversalIndex ];

    int n;
    for( i = 0; i < 8; i++){
      //cerr<<"Child = "<<traversal[i]<<endl;
      n = traversal[i];
      if(n == 0 ||  n==2 ||  n==4 || n==6)
	drawTree( node->child( traversal[i]), true,  viewRay, st);
      else
	drawTree( node->child( traversal[i] ), false,  viewRay, st);	
    }
  }
}
  

void MultiBrick::drawTree( const VolumeOctree<Brick*>* node, bool useLevel,
			     const Ray&  viewRay, const SliceTable& st)
{

  int i;
  //double  ts[8];
  double tmin, tmax, dt;

  if ( node == NULL ) return;
   
  Brick* brick = (*node)(); // get the contents of the node


  if( node->getType() == VolumeOctree<Brick*>::LEAF ||
      (brick->getLevel() >= drawLevel && useLevel)) {
     st.getParameters( brick, tmin, tmax, dt );
  if(debug)
    cerr<<"drawing node "<< node->getId()<<" at level "<<brick->getLevel()<<endl;

  if(!mode) {
    brick->draw(viewRay, drawWireFrame, reload != 0,  widgetPoint );
  } else {
     brick->draw( viewRay, alpha, drawWireFrame, reload != 0, tmin, tmax, dt);
  }  
  if(reload !=0)
    reload = 0;
/*     for( i = 0; i < 8; i++) */
/*       ts[i] = intersectParam(-viewRay.direction(), */
/* 			     brick->getCorner(i), viewRay); */
/*     sortParameters(ts,8); */
/*     brick->draw( viewRay, alpha, drawWireFrame, ts[7], ts[0], (ts[0]-ts[7])/slices ); */

  } else {
    int *traversal;
    int traversalIndex, x, y, z;
    Point min, max, mid;
    const VolumeOctree<Brick*>* child;
    child = node->child(0);
    
    mid = child->getMax();
    min = child->getMin();
    if(debug){
      cerr<<"PARENT node "<< node->getId()<<endl;
      //cerr<<"child 0: min,  max = "<<min<<", "<<mid<<endl;
    }


    if( viewRay.origin().x() < mid.x()) x = 0;
    else if( viewRay.origin().x() == mid.x()) x = 1;
    else x = 2;
    if( viewRay.origin().y() < mid.y()) y = 0;
    else if( viewRay.origin().y() == mid.y()) y = 1;
    else y = 2;
    if( viewRay.origin().z() < mid.z()) z = 0;
    else if( viewRay.origin().z() == mid.z()) z = 1;
    else z = 2;
    
    traversalIndex = 9*x + 3*y + z;
    if(debug){
      cerr<<"Traversal index = "<<traversalIndex<<endl;
    }

    traversal = traversalTable[ traversalIndex ];

    for( i = 0; i < 8; i++){
      //cerr<<"Child = "<<traversal[i]<<endl;
       drawTree( node->child( traversal[i] ), useLevel, viewRay, st);
    }
  }
}



#define MULTIBRICK_VERSION 1

void MultiBrick::io(Piostream&)
{
    // Nothing for now...
  NOT_FINISHED("MultiBrick::io");
}

bool
MultiBrick::saveobj(std::ostream&, const clString& format, GeomSave*)
{
   NOT_FINISHED("MultiBrick::saveobj");
    return false;
}

  
} // namespace SCICore
} // namespace GeomSpace
