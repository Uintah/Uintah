#ifndef GLTEXTURE3D_H
#define GLTEXTURE3D_H

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Containers/Array3.h>
#include <SCICore/Datatypes/ScalarFieldRGuchar.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/BBox.h>
#include "Octree.h"
#include <iostream>
namespace Kurt {
namespace Datatypes {

using SCICore::Datatypes::ScalarFieldRGuchar;
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
  GLTexture3D(ScalarFieldRGuchar *tex);
  //////////
  // Constructor
  GLTexture3D();
  // GROUP: Destructors
  //////////
  // Destructor
  ~GLTexture3D(){}
 
  // GROUP: Modify
  //////////  
  // Set a new scalarField
  void SetField( ScalarFieldRGuchar *tex);
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
  

  // GROUP: io
  /////////
  // Persistant representation
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

private:

  Octree<Brick*>* bontree;
  ScalarFieldRGuchar *tex;
  int levels;
  Point minP;
  Point maxP;
  int X, Y, Z;
  //  int maxBrick;  // max brick dimension
  int xmax, ymax, zmax;
  double dx,dy,dz;

  void computeTreeDepth();
  bool SetMaxBrickSize(int maxBrick);
  Octree<Brick*>* buildBonTree(Point min, Point max,
				     int xoff, int yoff, int zoff,
				     int xsize, int ysize, int zsize,
				     int level, Octree<Brick*>* parent);
  
  void BuildChild(int i, Point min, Point mid, Point max,
		  int xoff, int yoff, int zoff,
		  int xsize, int ysize, int zsize,
		  int X2, int Y2, int Z2,
		  int level,  Octree<Brick*>* node);
  
  void makeBrickData(int newx, int newy, int newz,
		     int xsize, int ysize, int zsize,
		     int xoff, int yoff, int zoff,
		     Array3<unsigned char>*& bd);

  void makeLowResBrickData(int xmax, int ymax, int zmax,
			   int xsize, int ysize, int zsize,
			   int xoff, int yoff, int zoff,
			   int level, int& padx, int& pady,
			   int& padz, Array3<unsigned char>*& bd);



  
  
};

} // end namespace Datatypes
} // end namespace Kurt
#endif
