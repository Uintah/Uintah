#ifndef GLTEXTURE3D_H
#define GLTEXTURE3D_H

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Containers/Array3.h>
#include <SCICore/Datatypes/ScalarFieldRGBase.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Thread/Runnable.h>
#include "Octree.h"
#include <iostream>
#include <deque>
using std::deque;

namespace SCICore{
namespace GeomSpace{
    class GLVolRenState;
    class FullRes;
    class ROI;
    class LOS;
  }
namespace Thread{
  class Semaphore;
  class ThreadGroup;
  }
}

namespace Kurt {
namespace Datatypes {

using SCICore::Datatypes::ScalarFieldRGBase;
using SCICore::Datatypes::Datatype;
using SCICore::Containers::LockingHandle;
using SCICore::Containers::Array3;
using SCICore::Geometry::Point;
using SCICore::Geometry::BBox;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
using SCICore::Thread::Semaphore;
using SCICore::Thread::Runnable;
using SCICore::Thread::ThreadGroup;


class Brick;
class GLTextureIterator;
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
class GLTexture3D;



typedef LockingHandle<GLTexture3D> GLTexture3DHandle;

class GLTexture3D : public Datatype {
  friend class GLTextureIterator;
  friend class SCICore::GeomSpace::GLVolRenState;
  friend class FullResIterator;
  friend class SCICore::GeomSpace::FullRes;
  friend class LOSIterator;
  friend class SCICore::GeomSpace::LOS;
  friend class ROIIterator;
  friend class SCICore::GeomSpace::ROI;

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

  // GROUP: io
  /////////
  // Persistant representation
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  bool CC() const {return isCC;}
  void get_minmax( double& min, double& max) const { min = _min, max = _max;}
  double SETVAL(double);

private:


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
				 Semaphore* thread_sema, 
				 Semaphore* total_threads, 
				 int& numTotal);
  template <class T>
    void BuildChild(int i, Point min, Point mid, Point max,
		    int xoff, int yoff, int zoff,
		    int xsize, int ysize, int zsize,
		    int X2, int Y2, int Z2,
		    int level,  T* tex, Octree<Brick*>* node,
		    Semaphore* thread_sema, Semaphore* total_threads,
		    int& numTotal);
  
//   template <class T>
//     void makeBrickData(int newx, int newy, int newz,
// 		       int xsize, int ysize, int zsize,
// 		       int xoff, int yoff, int zoff,
// 		       T* tex, Array3<unsigned char>*& bd);
  
//   template <class T>
//     void makeLowResBrickData(int xmax, int ymax, int zmax,
// 			     int xsize, int ysize, int zsize,
// 			     int xoff, int yoff, int zoff,
// 			     int level, int& padx, int& pady,
// 			     int& padz, T* tex,
// 			     Array3<unsigned char>*& bd);

  template <class T>
  class run_makeBrickData : public Runnable {
  public:
    run_makeBrickData(GLTexture3D* tex3D,
		      Semaphore *thread, Semaphore *total,
		      int newx, int newy, int newz,
		      int xsize, int ysize, int zsize,
		      int xoff, int yoff, int zoff, T *tex,
		      Array3<unsigned char>*& bd);
    virtual void run();
  private:
    GLTexture3D *tex3D;
    Semaphore *thread_sema, *total_threads;
    int newx, newy, newz;
    int xsize, ysize, zsize;
    int xoff, yoff, zoff;
    T* tex;
    Array3<unsigned char>* bd;
  };

  template <class T>
  class run_makeLowResBrickData : public Runnable {
  public:
    run_makeLowResBrickData(GLTexture3D* tex3D,
			    Semaphore *thread, Semaphore *total,
			    int xmax, int ymax, int zmax,
			    int xsize, int ysize, int zsize,
			    int xoff, int yoff, int zoff,
			    int& padx, int& pady, int& padz,
			    int level, T* tex, Array3<unsigned char>*& bd);
    virtual void run();
  private:
    GLTexture3D *tex3D;
    Semaphore *thread_sema, *total_threads;
    int xmax, ymax, zmax;
    int xsize, ysize, zsize;
    int xoff, yoff, zoff;
    int padx, pady, padz;
    int level;
    T* tex;
    Array3<unsigned char>* bd;
  };

  
};

} // end namespace Datatypes
} // end namespace Kurt
#endif
