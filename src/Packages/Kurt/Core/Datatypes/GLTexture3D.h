#ifndef GLTEXTURE3D_H
#define GLTEXTURE3D_H

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Array3.h>
#include <Core/Datatypes/ScalarFieldRGBase.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>
#include <Core/Thread/Runnable.h>
#include "Octree.h"
#include <iostream>
#include <deque>
using std::deque;

namespace SCIRun{
  class Semaphore;
  class ThreadGroup;
} // end namespace SCIRun

namespace Kurt {

using namespace SCIRun;

class GLVolRenState;
class FullRes;
class ROI;
class LOS;
class Brick;
class GLTextureIterator;

class GLTexture3D;
typedef LockingHandle<GLTexture3D> GLTexture3DHandle;

/**************************************

CLASS
   GLTexture3D
   
   Simple GLTexture3D Class.

GENERAL INFORMATION

   GLTexture3D.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Texture

DESCRIPTION
   GLTexture3D class.
  
WARNING
  
****************************************/

class GLTexture3D : public Datatype {

public:
  // GROUP: Constructors:
  //////////
  // Constructor
  GLTexture3D(ScalarFieldRGBase *tex);
  //////////
  // Constructor
  GLTexture3D();
  // GROUP: Destructors
  //////////
  // Destructor
  ~GLTexture3D();
 
  // GROUP: Modify
  //////////  
  // Set a new scalarField
  void SetField( ScalarFieldRGBase *tex);
  //////////
  // Change the BrickSize
  bool SetBrickSize( int brickSize );
  

  // GROUP: Access
  //////////
  // get min point
  const Point& min() const { return minP;}
  //////////
  // get max point
  const Point& max() const { return maxP;}
  /////////
  // the depth of the bontree
  int depth() const { return levels; }
  /////////
  // the depth of the bontree
  void get_bounds(BBox& b) const { b.extend(minP); b.extend(maxP);}
  /////////
  // Get the brick
  int getBrickSize(){ return xmax; }
  /////////
  // Get field size
  ScalarFieldRGBase* getField(){ return _tex; }
  /////////
  // Get the Bontree root
  const Octree<Brick*>* getBonTree() const {return bontree;}

  // GROUP: io
  /////////
  // Persistant representation
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  bool CC() const {return isCC;}
  void get_minmax( double& min, double& max) const { min = _min, max = _max;}
  double SETVAL(double);

private:
  static int max_workers;
  Octree<Brick*>* bontree;
  ScalarFieldRGBase *_tex;
  ThreadGroup *tg;
  int levels;
  Point minP;
  Point maxP;
  double _min, _max;
  int X, Y, Z;
  //  int maxBrick;  // max brick dimension
  int xmax, ymax, zmax;
  double dx,dy,dz;
  bool isCC;
  void SetBounds();
  void computeTreeDepth();
  void BuildTexture();
  bool SetMaxBrickSize(int maxBrick);

  template <class T>
    Octree<Brick*>* buildBonTree(Point min, Point max,
				 int xoff, int yoff, int zoff,
				 int xsize, int ysize, int zsize,
				 int level, T *tex,
				 Octree<Brick*>* parent,
				 Semaphore* thread_sema, ThreadGroup* tg);
  template <class T>
    void BuildChild(int i, Point min, Point mid, Point max,
		    int xoff, int yoff, int zoff,
		    int xsize, int ysize, int zsize,
		    int X2, int Y2, int Z2,
		    int level,  T* tex, Octree<Brick*>* node,
		    Semaphore* thread_sema, ThreadGroup* tg);
  
  template <class T>
  class run_makeBrickData : public Runnable {
  public:
    run_makeBrickData(GLTexture3D* tex3D,
		      Semaphore *thread,
		      int newx, int newy, int newz,
		      int xsize, int ysize, int zsize,
		      int xoff, int yoff, int zoff, T *tex,
		      Array3<unsigned char>*& bd);
    virtual void run();
  private:
    GLTexture3D *tex3D;
    Semaphore *thread_sema;
    int newx, newy, newz;
    int xsize, ysize, zsize;
    int xoff, yoff, zoff;
    T* tex;
    Array3<unsigned char>* bd;
  };

  //  template <class T>
  class run_makeLowResBrickData : public Runnable {
  public:
    run_makeLowResBrickData(GLTexture3D* tex3D,
			    Semaphore *thread,
			    int xmax, int ymax, int zmax,
			    int xsize, int ysize, int zsize,
			    int xoff, int yoff, int zoff,
			    int& padx, int& pady, int& padz,
			    int level, Octree<Brick*>* node,
			    Array3<unsigned char>*& bd);
    virtual void run();
  private:
    GLTexture3D* tex3D;
    Octree<Brick*>* parent;
    Semaphore *thread_sema;
    int xmax, ymax, zmax;
    int xsize, ysize, zsize;
    int xoff, yoff, zoff;
    int padx, pady, padz;
    int level;
    //    T* tex;
    Array3<unsigned char>* bd;
  };

  
};



template <class T>
Octree<Brick*>*
GLTexture3D::buildBonTree(Point min, Point max,
            int xoff, int yoff, int zoff,
            int xsize, int ysize, int zsize,
	    int level, T *tex, Octree<Brick*>* parent,
	    Semaphore* thread_sema, ThreadGroup *tg)
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
    brickData->newsize( newz, newy, newx);

//     GLTexture3D::run_makeBrickData<T> mbd(this, 
// 					  thread_sema,
// 					  newx,newy,newz,
// 					  xsize,ysize,zsize,
// 					  xoff,yoff,zoff,
// 					  tex, brickData);
//     mbd.run();

    thread_sema->down();

    Thread *t = new Thread(new GLTexture3D::run_makeBrickData<T>(this, 
					 thread_sema, 
					 newx,newy,newz,
					 xsize,ysize,zsize,
					 xoff,yoff,zoff,
					 tex, brickData),
			   "makeBrickData worker",tg);


    brick = scinew Brick(min, max, padx,  pady, padz, level, brickData);

    node = scinew Octree<Brick*>(brick, Octree<Brick *>::LEAF,
				    parent );
  } else { // we must subdivide

    brickData->newsize( zmax, ymax, xmax);
    double stepx, stepy, stepz;
    if( level > 0 ){

      stepx = pow(2.0, levels - level);
      if( xmax > xsize ) {
	padx=(xmax - xsize)*stepx;
      } else {
	if( xmax * stepx > xsize){
	  padx = (xmax*stepx - xsize)/stepx;
	}
      }
      stepy = pow(2.0, levels - level);
      if( ymax > ysize ) {
	pady = (ymax - ysize)*stepy;
      } else {
	if( ymax * stepy > ysize){
	  pady = (ymax*stepy - ysize)/stepy;
	}
      }
      stepz = pow(2.0, levels - level);
      if( zmax > zsize ) {
	stepz = 1; padz = (zmax - zsize)*stepz;
      } else {
	if( zmax * stepz > zsize){
	  padz = (zmax*stepz - zsize)/stepz;
	}
      }
    }

    clString  group_name( "thread group ");
    group_name = group_name + to_string( level )();
    ThreadGroup *group = scinew ThreadGroup( group_name() );
    
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
			  thread_sema, group);
      }
    } else if( Z2 > Y2 && Z2 > X2 ) {
      mid = min + Vector(diag.x(),
			 diag.y(),
			 dz*(sz-1));
      
      BuildChild(0, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, ysize, sz, level, tex, node, 
			  thread_sema, group);
      BuildChild(1, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, ysize, sz, level, tex, node, 
			  thread_sema, group);
    } else  if( Y2 > Z2 && Y2 > X2 ) {
      mid = min + Vector(diag.x(),
			 dy*(sy - 1),
			 diag.z());
      BuildChild(0, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, zsize, level, tex, node, 
			  thread_sema, group);
      BuildChild(2, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, zsize, level, tex, node, 
			  thread_sema, group);
    } else  if( X2 > Z2 && X2 > Y2 ) {
      mid = min + Vector(dx*(sx-1),
			 diag.y(),
			 diag.z());
      BuildChild(0, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, zsize, level, tex, node, 
			  thread_sema, group);
      BuildChild(4,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, zsize, level, tex, node, 
			  thread_sema, group);
    } else if( Z2 == Y2 ){
      mid = min + Vector(diag.x(),
			 dy * (sy - 1),
			 dz* (sz - 1));
      BuildChild(0,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level, tex, node, 
			  thread_sema, group);
      BuildChild(1,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level, tex, node, 
			  thread_sema, group);
      BuildChild(2,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level, tex, node, 
			  thread_sema, group);
      BuildChild(3,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level, tex, node, 
			  thread_sema, group);
    } else if( X2 == Y2 ){
      mid = min + Vector(dx*(sx - 1), dy*(sy-1),
			 diag.z());
      BuildChild(0,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level, tex, node, 
			  thread_sema, group);
      BuildChild(2,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level, tex, node, 
			  thread_sema, group);
      BuildChild(4,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level, tex, node, 
			  thread_sema, group);
      BuildChild(6,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level, tex, node, 
			  thread_sema, group);
    } else if( Z2 == X2 ){
      mid = min + Vector(dx*(sx-1),
			 diag.y(),
			 dz*(sz-1));
      BuildChild(0,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level, tex, node, 
			  thread_sema, group);
      BuildChild(1,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level, tex, node, 
			  thread_sema, group);
      BuildChild(4,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level, tex, node, 
			  thread_sema, group);
      BuildChild(5,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level, tex, node, 
			  thread_sema, group);
    }

    group->join();
    delete group;
      
    thread_sema->down();
    Thread *t = new Thread(new GLTexture3D::run_makeLowResBrickData(this, 
					   thread_sema,
					   xmax, ymax, zmax,
					   xsize, ysize, zsize,
					   xoff, yoff, zoff, 
					   padx, pady, padz,
					   level, node, brickData),
			   "makeLowResBrickData worker", tg);


  }
  return node;
}

template <class T>
void GLTexture3D::BuildChild(int i, Point min, Point mid, Point max,
			     int xoff, int yoff, int zoff,
			     int xsize, int ysize, int zsize,
			     int X2, int Y2, int Z2,
			     int level,  T* tex, Octree<Brick*>* node,
			     Semaphore* thread_sema, ThreadGroup *tg)
{
  Point pmin, pmax;

  switch( i ) {
  case 0:
    pmin = min;
    pmax = mid;
    node->SetChild(0, buildBonTree(pmin, pmax, xoff, yoff, zoff,
				   X2, Y2, Z2, level, tex, node, 
			   thread_sema, tg));
    break;
  case 1:
    pmin = min;
    pmax = mid;
    pmin.z(mid.z());
    pmax.z(max.z());
    node->SetChild(1, buildBonTree(pmin, pmax,
				   xoff, yoff, zoff + Z2 -1,
				   X2, Y2, zsize-Z2+1, level, tex, node, 
				   thread_sema, tg));
    break;
  case 2:
    pmin = min;
    pmax = mid;
    pmin.y(mid.y());
    pmax.y(max.y());
    node->SetChild(2, buildBonTree(pmin, pmax,
				   xoff, yoff + Y2 - 1, zoff,
				   X2, ysize - Y2 + 1, Z2, level, tex, node, 
				   thread_sema, tg));
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
				   thread_sema, tg));
    break;
  case 4:
    pmin = min;
    pmax = mid;
    pmin.x(mid.x());
    pmax.x(max.x());
    node->SetChild(4, buildBonTree(pmin, pmax,
				   xoff + X2 - 1, yoff, zoff,
				   xsize - X2 + 1, Y2, Z2, level, tex, node, 
				   thread_sema, tg));
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
				   thread_sema, tg));
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
				   thread_sema, tg));
    break;
  case 7:
   pmin = mid;
   pmax = max;
   node->SetChild(7, buildBonTree(pmin, pmax,  xoff + X2 - 1,
				  yoff + Y2 - 1, zoff +  Z2 - 1,
				  xsize - X2 + 1, ysize - Y2 + 1,
				  zsize - Z2 + 1, level, tex, node, 
				  thread_sema, tg));
   break;
  default:
    break;
  }
}


template <class T>	
void					
GLTexture3D::run_makeBrickData<T>::run() 
{
  int i,j,k,ii,jj,kk;

  for(kk = 0, k = zoff; kk < zsize; kk++, k++)
    for(jj = 0, j = yoff; jj < ysize; jj++, j++)
      for(ii = 0, i = xoff; ii < xsize; ii++, i++){
	(*bd)(kk,jj,ii) = tex3D->SETVAL( tex->grid(i,j,k) );
  }
  thread_sema->up();
}

} // End namespace Kurt

#endif
