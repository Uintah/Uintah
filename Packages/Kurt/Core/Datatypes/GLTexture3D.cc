#include "GLTexture3D.h"
#include "Brick.h"
#include "VolumeUtils.h"
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Persistent/Persistent.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Datatypes/ScalarFieldRGint.h>
#include <SCICore/Datatypes/ScalarFieldRGshort.h>
#include <SCICore/Datatypes/ScalarFieldRGfloat.h>
#include <SCICore/Datatypes/ScalarFieldRGdouble.h>
#include <SCICore/Datatypes/ScalarFieldRGuchar.h>
#include <SCICore/Datatypes/ScalarFieldRGchar.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Thread/Thread.h>
#include <SCICore/Thread/Semaphore.h>
#include <SCICore/Thread/ThreadGroup.h>
#include <Uintah/Datatypes/NCScalarField.h>
#include <Uintah/Datatypes/CCScalarField.h>


#include <GL/gl.h>
#include <iostream>
#include <string>
#include <deque>

using std::cerr;
using std::endl;
using std::string;
using std::deque;


namespace Kurt {
namespace Datatypes {

using namespace SCICore::Datatypes;
using namespace SCICore::Thread;
using SCICore::Containers::clString;
using SCICore::Math::Max;

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
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;
    using SCICore::Geometry::Pio;
    NOT_FINISHED("GLTexture3D::io(Piostream&)");
}

GLTexture3D::GLTexture3D() :
  _tex(0), X(0), Y(0),
  Z(0), xmax(0), ymax(0), zmax(0), isCC(false),
  tg(new ThreadGroup("texture group"))
{
}

GLTexture3D::GLTexture3D(ScalarFieldRGBase *tex ) :
  _tex(tex), X(tex->nx), Y(tex->ny),
  Z(tex->nz),  xmax(128), ymax(128), zmax(128), isCC(false),
  tg(new ThreadGroup("texture group"))
{
  tex->get_bounds( minP, maxP );
  tex->get_minmax( _min, _max );
  SetBounds();
  computeTreeDepth(); 
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
  cerr<<"Type = "<<_tex->getType()<<endl;
  Semaphore* thread_sema = new Semaphore( "worker count semhpore",
					  Max(Thread::numProcessors(), 8));
  Semaphore* total_threads = new Semaphore("total workers semaphore", 0);
  int numTotal=0;
  if( _tex->getRGDouble() ){
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,
			   _tex->getRGDouble(), 0, thread_sema, 
			   total_threads, numTotal);
  } else if( _tex->getRGFloat() ){
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,
			   _tex->getRGFloat(), 0, thread_sema, 
			   total_threads, numTotal);
  } else if( _tex->getRGInt() ) {
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,
			   _tex->getRGInt(), 0, thread_sema, 
			   total_threads, numTotal);
  } else if( _tex->getRGShort() ){
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,
			   _tex->getRGShort(), 0, thread_sema, 
			   total_threads, numTotal);
  } else if( _tex->getRGUchar() ){
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,
			   _tex->getRGUchar(), 0, thread_sema, 
			   total_threads, numTotal);
  } else if( _tex->getRGChar() ){
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,
			   _tex->getRGChar(), 0, thread_sema, 
			   total_threads, numTotal);
  } else if(NCScalarField<double> *sfd =
	    dynamic_cast<NCScalarField<double> *> (_tex)){
    cerr<<"Type = <NCScalarField<double>\n";
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, sfd, 0, 
			   thread_sema, total_threads, numTotal);
  } else if(NCScalarField<int> *sfi =
	    dynamic_cast<NCScalarField<int> *> (_tex)){
    cerr<<"Type = NCScalarField<int>\n";
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, sfi, 0, 
			   thread_sema, total_threads, numTotal);
  } else if(NCScalarField<long> *sfl =
	    dynamic_cast<NCScalarField<long> *> (_tex)){
    cerr<<"Type = NCScalarField<long>\n";
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, sfl, 0, 
			   thread_sema, total_threads, numTotal);
  } else if(CCScalarField<double> *sfd =
	    dynamic_cast<CCScalarField<double> *> (_tex)){
    cerr<<"Type = CCScalarField<double>\n";
    isCC = true;
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, sfd, 0, 
			   thread_sema, total_threads, numTotal);
  } else if(CCScalarField<int> *sfi =
	    dynamic_cast<CCScalarField<int> *> (_tex)) {
    cerr<<"Type = CCScalarField<int>\n";
    isCC = true;
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, sfi, 0, 
			   thread_sema, total_threads, numTotal);
  } else if(CCScalarField<long> *sfl =
	    dynamic_cast<CCScalarField<long> *> (_tex)) {
    cerr<<"Type = CCScalarField<long>\n";
    isCC = true;
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, sfl, 0, 
			   thread_sema, total_threads, numTotal);
  } else {
    cerr<<"Error: cast didn't work!\n";
  }
//   total_threads->down(numTotal);
  ASSERT(bontree != 0x0);
}

bool
GLTexture3D::SetBrickSize(int bsize)
{
  xmax = ymax = zmax = bsize;
  X = _tex->nx;
  Y = _tex->ny;
  Z = _tex->nz;
  
  if( bontree ) delete bontree;
  computeTreeDepth();
  BuildTexture();
  return true;
}

void
GLTexture3D::SetField( ScalarFieldRGBase *tex )
{
  this->_tex = tex;
  X = tex->nx;
  Y = tex->ny;
  Z = tex->nz;

  int size = std::max(X,Y);
  size = std::max(size,Z);

  tex->get_bounds( minP, maxP );
  tex->get_minmax( _min, _max );
  xmax = ymax = zmax = 128;


  SetBounds();
  computeTreeDepth(); 
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

  levels = (( xdepth > ydepth)? (( xdepth > zdepth)? xdepth: zdepth):
	    (( ydepth > zdepth)? ydepth: zdepth));
  
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
	    int level, T *tex, Octree<Brick*>* parent,
	    Semaphore* thread_sema, Semaphore* total_threads, int& numTotal)
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
    return node;
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
    // set the pad size for each direction
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

//     thread_sema->down();
//     numTotal++;
//     Thread *t = new Thread(new GLTexture3D::run_makeBrickData<T>(this, 
// 				     thread_sema, total_threads, 
// 				     newx,newy,newz,
// 				     xsize,ysize,zsize,
// 				     xoff,yoff,zoff,
// 				     tex, brickData),
// 			   "makeBrickData worker",tg);
//     t->detach();
    
    GLTexture3D::run_makeBrickData<T> mbd(this, 
					  thread_sema, total_threads, 
					  newx,newy,newz,
					  xsize,ysize,zsize,
					  xoff,yoff,zoff,
					  tex, brickData);
    mbd.run();

    brick = scinew Brick(min, max, padx,  pady, padz, level, brickData);

    node = scinew Octree<Brick*>(brick, Octree<Brick *>::LEAF,
				    parent );
  } else { // we must subdivide
//     thread_sema->down();
//     numTotal++;
//     Thread *t = new Thread(new GLTexture3D::run_makeLowResBrickData<T>(this, 
// 					   thread_sema, total_threads,
// 					   xmax, ymax, zmax,
// 					   xsize, ysize, zsize,
// 					   xoff, yoff, zoff, 
// 					   padx, pady, padz,
// 					   level, tex, brickData),
// 			   "makeLowResBrickData worker", tg);
//     t->detach();

    GLTexture3D::run_makeLowResBrickData<T> mlrbd(this, 
						  thread_sema, total_threads,
						  xmax, ymax, zmax,
						  xsize, ysize, zsize,
						  xoff, yoff, zoff, 
						  padx, pady, padz,
						  level, tex, brickData);
    mlrbd.run();

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
		    xsize, ysize, zsize, sx, sy, sz,level,tex, node, 
			   thread_sema, total_threads, numTotal);
      }
    } else if( Z2 > Y2 && Z2 > X2 ) {
      mid = min + Vector(diag.x(),
			 diag.y(),
			 dz*(sz-1));
      
      BuildChild(0, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, ysize, sz, level, tex, node, 
			   thread_sema, total_threads, numTotal);
      BuildChild(1, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, ysize, sz, level, tex, node, 
			   thread_sema, total_threads, numTotal);
    } else  if( Y2 > Z2 && Y2 > X2 ) {
      mid = min + Vector(diag.x(),
			 dy*(sy - 1),
			 diag.z());
      BuildChild(0, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, zsize, level, tex, node, 
			   thread_sema, total_threads, numTotal);
      BuildChild(2, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, zsize, level, tex, node, 
			   thread_sema, total_threads, numTotal);
    } else  if( X2 > Z2 && X2 > Y2 ) {
      mid = min + Vector(dx*(sx-1),
			 diag.y(),
			 diag.z());
      BuildChild(0, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, zsize, level, tex, node, 
			   thread_sema, total_threads, numTotal);
      BuildChild(4,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, zsize, level, tex, node, 
			   thread_sema, total_threads, numTotal);
    } else if( Z2 == Y2 ){
      mid = min + Vector(diag.x(),
			 dy * (sy - 1),
			 dz* (sz - 1));
      BuildChild(0,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level, tex, node, 
			   thread_sema, total_threads, numTotal);
      BuildChild(1,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level, tex, node, 
			   thread_sema, total_threads, numTotal);
      BuildChild(2,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level, tex, node, 
			   thread_sema, total_threads, numTotal);
      BuildChild(3,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level, tex, node, 
			   thread_sema, total_threads, numTotal);
    } else if( X2 == Y2 ){
      mid = min + Vector(dx*(sx - 1), dy*(sy-1),
			 diag.z());
      BuildChild(0,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level, tex, node, 
			   thread_sema, total_threads, numTotal);
      BuildChild(2,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level, tex, node, 
			   thread_sema, total_threads, numTotal);
      BuildChild(4,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level, tex, node, 
			   thread_sema, total_threads, numTotal);
      BuildChild(6,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level, tex, node, 
			   thread_sema, total_threads, numTotal);
    } else if( Z2 == X2 ){
      mid = min + Vector(dx*(sx-1),
			 diag.y(),
			 dz*(sz-1));
      BuildChild(0,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level, tex, node, 
			   thread_sema, total_threads, numTotal);
      BuildChild(1,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level, tex, node, 
			   thread_sema, total_threads, numTotal);
      BuildChild(4,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level, tex, node, 
			   thread_sema, total_threads, numTotal);
      BuildChild(5,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level, tex, node, 
			   thread_sema, total_threads, numTotal);
    }
  }
  return node;
}

template <class T>
void GLTexture3D::BuildChild(int i, Point min, Point mid, Point max,
			     int xoff, int yoff, int zoff,
			     int xsize, int ysize, int zsize,
			     int X2, int Y2, int Z2,
			     int level,  T* tex, Octree<Brick*>* node,
			     Semaphore* thread_sema, Semaphore* total_threads,
			     int& numTotal)
{
  Point pmin, pmax;

  switch( i ) {
  case 0:
    pmin = min;
    pmax = mid;
    node->SetChild(0, buildBonTree(pmin, pmax, xoff, yoff, zoff,
				   X2, Y2, Z2, level, tex, node, 
			   thread_sema, total_threads, numTotal));
    break;
  case 1:
    pmin = min;
    pmax = mid;
    pmin.z(mid.z());
    pmax.z(max.z());
    node->SetChild(1, buildBonTree(pmin, pmax,
				   xoff, yoff, zoff + Z2 -1,
				   X2, Y2, zsize-Z2+1, level, tex, node, 
				   thread_sema, total_threads, numTotal));
    break;
  case 2:
    pmin = min;
    pmax = mid;
    pmin.y(mid.y());
    pmax.y(max.y());
    node->SetChild(2, buildBonTree(pmin, pmax,
				   xoff, yoff + Y2 - 1, zoff,
				   X2, ysize - Y2 + 1, Z2, level, tex, node, 
				   thread_sema, total_threads, numTotal));
    break;
  case 3:
    pmin = mid;
    pmax = max;
    pmin.x(min.x());
    pmax.x(mid.x());
    node->SetChild(3, buildBonTree(pmin, pmax,
				   xoff, yoff + Y2 - 1 , zoff + Z2 - 1,
				   X2, ysize - Y2 + 1, zsize - Z2 + 1, level, 
				   tex, node, 
				   thread_sema, total_threads, numTotal));
    break;
  case 4:
    pmin = min;
    pmax = mid;
    pmin.x(mid.x());
    pmax.x(max.x());
    node->SetChild(4, buildBonTree(pmin, pmax,
				   xoff + X2 - 1, yoff, zoff,
				   xsize - X2 + 1, Y2, Z2, level, tex, node, 
				   thread_sema, total_threads, numTotal));
    break;
  case 5:
    pmin = mid;
    pmax = max;
    pmin.y(min.y());
    pmax.y(mid.y());
    node->SetChild(5, buildBonTree(pmin, pmax,
				   xoff + X2 - 1, yoff, zoff +  Z2 - 1,
				   xsize - X2 + 1, Y2, zsize - Z2 + 1, level, 
				   tex, node, 
				   thread_sema, total_threads, numTotal));
    break;
  case 6:
    pmin = mid;
    pmax = max;
    pmin.z(min.z());
    pmax.z(mid.z());
    node->SetChild(6, buildBonTree(pmin, pmax,
				   xoff + X2 - 1, yoff + Y2 - 1, zoff,
				   xsize - X2 + 1, ysize - Y2 + 1, Z2, level, 
				   tex, node, 
				   thread_sema, total_threads, numTotal));
    break;
  case 7:
   pmin = mid;
   pmax = max;
   node->SetChild(7, buildBonTree(pmin, pmax,  xoff + X2 - 1,
				  yoff + Y2 - 1, zoff +  Z2 - 1,
				  xsize - X2 + 1, ysize - Y2 + 1,
				  zsize - Z2 + 1, level, tex, node, 
				  thread_sema, total_threads, numTotal));
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
GLTexture3D::run_makeBrickData<T>::run_makeBrickData(
				GLTexture3D* tex3D,
			        Semaphore *thread, Semaphore *total,
				int newx, int newy, int newz,
				int xsize, int ysize, int zsize,
				int xoff, int yoff, int zoff, T *tex,
				Array3<unsigned char>*& bd) :
  tex3D(tex3D),
  thread_sema( thread ), total_threads( total ),
  newx(newx), newy(newy), newz(newz),
  xsize(xsize), ysize(ysize), zsize(zsize),
  xoff(xoff), yoff(yoff), zoff(zoff),
  tex(tex), bd(bd)
{
  // constructor
}


template <class T>	
void					
GLTexture3D::run_makeBrickData<T>::run() 
{
  int i,j,k,ii,jj,kk;

  bd->newsize( newz, newy, newx);
  for(kk = 0, k = zoff; kk < zsize; kk++, k++)
    for(jj = 0, j = yoff; jj < ysize; jj++, j++)
      for(ii = 0, i = xoff; ii < xsize; ii++, i++){
	(*bd)(kk,jj,ii) = tex3D->SETVAL( tex->grid(i,j,k) );
  }
  thread_sema->up();
  total_threads->up();

}

template <class T>						
GLTexture3D::run_makeLowResBrickData<T>::run_makeLowResBrickData(
				      GLTexture3D* tex3D,
			              Semaphore *thread, Semaphore *total,
				      int xmax, int ymax, int zmax,
				      int xsize, int ysize, int zsize,
				      int xoff, int yoff, int zoff,
				      int& padx, int& pady, int& padz,
				      int level, T* tex,
				      Array3<unsigned char>*& bd) :
  tex3D(tex3D),
  thread_sema( thread ), total_threads( total ),
  xmax(xmax), ymax(ymax), zmax(zmax),
  xsize(xsize), ysize(ysize), zsize(zsize),
  xoff(xoff), yoff(yoff), zoff(zoff),
  padx(padx), pady(pady), padz(padz),
  level(level), tex(tex), bd(bd)
{
  // constructor
}

template <class T>
void
GLTexture3D::run_makeLowResBrickData<T>::run() 
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
	  k00 = Interpolate(tex3D->SETVAL( tex->grid(i,j,k) ),
			    tex3D->SETVAL( tex->grid(i,j,k1)),dk);
	  k01 = Interpolate(tex3D->SETVAL( tex->grid(i1,j,k)),
			    tex3D->SETVAL( tex->grid(i1,j,k1)),dk);
	  k10 = Interpolate(tex3D->SETVAL( tex->grid(i,j1,k)),
			    tex3D->SETVAL( tex->grid(i,j,k1)),dk);
	  k11 = Interpolate(tex3D->SETVAL( tex->grid(i1,j1,k)),
			    tex3D->SETVAL( tex->grid(i1,j1,k1)),dk);
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
      dx = pow(2.0, tex3D->depth() - level);
      if( xmax * dx > xsize){
	padx = (xmax*dx - xsize)/dx;
      }
    }
    if( ymax > ysize ) {
      dy = 1; pady = (ymax - ysize);
    } else {
      dy = pow(2.0, tex3D->depth() - level);
      if( ymax * dy > ysize){
	pady = (ymax*dy - ysize)/dy;
      }
    }
    if( zmax > zsize ) {
      dz = 1; padz = (zmax - zsize);
    } else {
      dz = pow(2.0, tex3D->depth() - level);
      if( zmax * dz > zsize){
	padz = (zmax*dz - zsize)/dz;
      }
    }
  }
  
  for(kk = 0, k = zoff; kk < zmax; kk++, k+=dz){
      for(jj = 0, j = yoff; jj < ymax; jj++, j+=dy){
	for(ii = 0, i = xoff; ii < xmax; ii++, i+=dx){
	  if( i < xoff + xsize && j < yoff + ysize && k < zoff + zsize){
	    (*bd)(kk,jj,ii) = tex3D->SETVAL( tex->grid(i,j,k) );
	  }
	}
      }
  }    
  thread_sema->up();
  total_threads->up();
}


} // end namespace Datatypes
} // end namespace Kurt
