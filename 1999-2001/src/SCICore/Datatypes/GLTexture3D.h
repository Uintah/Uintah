#ifndef GLTEXTURE3D_H
#define GLTEXTURE3D_H

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Containers/Array3.h>
#include <SCICore/Datatypes/ScalarFieldRGBase.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Datatypes/Octree.h>
#include <iostream>
#include <deque>
using std::deque;

namespace SCICore {
namespace Datatypes {

using SCICore::Datatypes::ScalarFieldRGBase;
using SCICore::Datatypes::Datatype;
using SCICore::Containers::LockingHandle;
using SCICore::Containers::Array3;
using SCICore::Geometry::Point;
using SCICore::Geometry::BBox;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;


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
  friend class FullResIterator;
  friend class LOSIterator;
  friend class ROIIterator;
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

private:


  Octree<Brick*>* bontree;
  ScalarFieldRGBase *_tex;
  int levels;
  Point minP;
  Point maxP;
  double _min, _max;
  int X, Y, Z;
  //  int maxBrick;  // max brick dimension
  int xmax, ymax, zmax;
  double dx,dy,dz;
  bool isCC;
  double SETVAL(double);
  void SetBounds();
  void computeTreeDepth();
  void BuildTexture();
  bool SetMaxBrickSize(int maxBrick);

  template <class T>
    Octree<Brick*>* buildBonTree(Point min, Point max,
				 int xoff, int yoff, int zoff,
				 int xsize, int ysize, int zsize,
				 int level, T *tex,
				 Octree<Brick*>* parent);
  template <class T>
    void BuildChild(int i, Point min, Point mid, Point max,
		    int xoff, int yoff, int zoff,
		    int xsize, int ysize, int zsize,
		    int X2, int Y2, int Z2,
		    int level,  T* tex, Octree<Brick*>* node);
  
  template <class T>
    void makeBrickData(int newx, int newy, int newz,
		       int xsize, int ysize, int zsize,
		       int xoff, int yoff, int zoff,
		       T* tex, Array3<unsigned char>*& bd);
  
  template <class T>
    void makeLowResBrickData(int xmax, int ymax, int zmax,
			     int xsize, int ysize, int zsize,
			     int xoff, int yoff, int zoff,
			     int level, int& padx, int& pady,
			     int& padz, T* tex,
			     Array3<unsigned char>*& bd);
  
};

} // end namespace Datatypes
} // end namespace SCICore
#endif
