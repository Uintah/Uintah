#include "GLTexture3D.h"
#include "Brick.h"
#include "VolumeUtils.h"
#include <Core/Util/NotFinished.h>
#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Containers/String.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/ThreadGroup.h>
#include <Packages/Uintah/Core/Datatypes/NCScalarField.h>
#include <Packages/Uintah/Core/Datatypes/CCScalarField.h>


#include <GL/gl.h>
#include <iostream>
#include <string>
#include <deque>

namespace Kurt {

using std::cerr;
using std::endl;
using std::string;
using std::deque;

using namespace SCIRun;
using namespace Uintah;

int GLTexture3D::max_workers = 0;

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

PersistentTypeID GLTexture3D::type_id("GLTexture3D", "Datatype", maker);
#define GLTexture3D_VERSION 3

void GLTexture3D::io(Piostream&)
{
  using namespace SCIRun;
  NOT_FINISHED("GLTexture3D::io(Piostream&)");
}

GLTexture3D::GLTexture3D() :
  _tex(0), X(0), Y(0),
  Z(0), xmax(0), ymax(0), zmax(0), isCC(false),
  tg(0)
{
}

GLTexture3D::GLTexture3D(ScalarFieldRGBase *tex ) :
  _tex(tex), X(tex->nx), Y(tex->ny),
  Z(tex->nz),  xmax(128), ymax(128), zmax(128), isCC(false),
  tg(0)
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
//   cerr<<"Type = "<<_tex->getType()<<endl;
  max_workers = Max(Thread::numProcessors()/2, 8);
  Semaphore* thread_sema = new Semaphore( "worker count semhpore",
					  max_workers);  

//   cerr<<"Max_worker threads = "<<max_workers<<endl;
  if( _tex->getRGDouble() ){
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,
			   _tex->getRGDouble(), 0, thread_sema, tg); 
			   
  } else if( _tex->getRGFloat() ){
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,
			   _tex->getRGFloat(), 0, thread_sema, tg); 
			   
  } else if( _tex->getRGInt() ) {
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,
			   _tex->getRGInt(), 0, thread_sema, tg); 
			   
  } else if( _tex->getRGShort() ){
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,
			   _tex->getRGShort(), 0, thread_sema, tg); 
			   
  } else if( _tex->getRGUchar() ){
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,
			   _tex->getRGUchar(), 0, thread_sema, tg); 

  } else if( _tex->getRGChar() ){
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0,
			   _tex->getRGChar(), 0, thread_sema, tg); 
			   
  } else if(NCScalarField<double> *sfd =
	    dynamic_cast<NCScalarField<double> *> (_tex)){
    cerr<<"Type = <NCScalarField<double>\n";
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, sfd, 0, 
			   thread_sema, tg);
  } else if(NCScalarField<int> *sfi =
	    dynamic_cast<NCScalarField<int> *> (_tex)){
    cerr<<"Type = NCScalarField<int>\n";
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, sfi, 0, 
			   thread_sema, tg);
  } else if(NCScalarField<long> *sfl =
	    dynamic_cast<NCScalarField<long> *> (_tex)){
    cerr<<"Type = NCScalarField<long>\n";
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, sfl, 0, 
			   thread_sema, tg);
  } else if(CCScalarField<double> *sfd =
	    dynamic_cast<CCScalarField<double> *> (_tex)){
    cerr<<"Type = CCScalarField<double>\n";
    isCC = true;
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, sfd, 0, 
			   thread_sema, tg);
  } else if(CCScalarField<int> *sfi =
	    dynamic_cast<CCScalarField<int> *> (_tex)) {
    cerr<<"Type = CCScalarField<int>\n";
    isCC = true;
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, sfi, 0, 
			   thread_sema, tg);
  } else if(CCScalarField<long> *sfl =
	    dynamic_cast<CCScalarField<long> *> (_tex)) {
    cerr<<"Type = CCScalarField<long>\n";
    isCC = true;
    bontree = buildBonTree(minP, maxP, 0, 0, 0, X, Y, Z, 0, sfl, 0, 
			   thread_sema, tg);
  } else {
    cerr<<"Error: cast didn't work!\n";
  }
  thread_sema->down(max_workers);
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
			        Semaphore *thread,
				int newx, int newy, int newz,
				int xsize, int ysize, int zsize,
				int xoff, int yoff, int zoff, T *tex,
				SCIRun::Array3<unsigned char>*& bd) :
  tex3D(tex3D),
  thread_sema( thread ),
  newx(newx), newy(newy), newz(newz),
  xsize(xsize), ysize(ysize), zsize(zsize),
  xoff(xoff), yoff(yoff), zoff(zoff),
  tex(tex), bd(bd)
{
  // constructor
}



GLTexture3D::run_makeLowResBrickData::run_makeLowResBrickData(
				      GLTexture3D* tex3D,
			              Semaphore *thread,
				      int xmax, int ymax, int zmax,
				      int xsize, int ysize, int zsize,
				      int xoff, int yoff, int zoff,
				      int& padx, int& pady, int& padz,
				      int level, Octree<Brick*>* node,
				      SCIRun::Array3<unsigned char>*& bd) :
  tex3D(tex3D),
  thread_sema( thread ), 
  xmax(xmax), ymax(ymax), zmax(zmax),
  xsize(xsize), ysize(ysize), zsize(zsize),
  xoff(xoff), yoff(yoff), zoff(zoff),
  padx(padx), pady(pady), padz(padz),
  level(level), parent(node), bd(bd)
{
  // constructor
}

void
GLTexture3D::run_makeLowResBrickData::run() 
{
  using namespace SCIRun;

  int  i,j,k;
  int ii,jj,kk;
  double dx, dy, dz;

  if( level == 1 ){
     thread_sema->up();
     return;
  } else {
    Brick *brick = 0;
    SCIRun::Array3<unsigned char>* brickTexture;
    int x,y,z;
    for( kk = 0, k = 0; kk < zmax; kk++, k+=2){
      if ( 2*kk >= zmax )  z = 1; else z = 0;
      if ( 2*kk == zmax ) k = 1;
      for( jj = 0, j = 0; jj < ymax; jj++, j+=2){
	if( 2*jj >= ymax) y = 2; else y = 0;
	if( 2*jj == ymax) j = 1;
	for (ii = 0, i = 0; ii < xmax; ii++, i+=2){
	  if( 2*ii >= xmax ) x = 4; else x = 0;
	  if( 2*ii == xmax ) i = 1;

	  brick = (*((*this->parent)[x+y+z]))();
	  brickTexture = brick->texture();

	  // This code does simple subsampling.  Uncomment the 
	  // center section to perform averaging.
	  if( brick == 0 ){
	    (*bd)(kk,jj,ii) = (unsigned char)0;
//////////// Uncomment for texel averageing
// 	  } else if((ii > 0 && ii < xmax - 1) &&
// 	     (jj > 0 && jj < ymax - 1) &&
// 	     (kk > 0 && kk < zmax - 1)){
// 	    (*bd)(kk,jj,ii) = (0.5*(*brickTexture)(k,j,i)           +
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
	    (*bd)(kk,jj,ii) = (*brickTexture)(k,j,i);
	  }
	}
      }
    }
  }    
  thread_sema->up();
}

} // End namespace Kurt

