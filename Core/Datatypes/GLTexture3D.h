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

#ifndef GLTEXTURE3D_H
#define GLTEXTURE3D_H

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Array3.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>
#include <Core/Datatypes/Octree.h>
#include <iostream>
#include <deque>
using std::deque;

namespace SCIRun {



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
  GLTexture3D(FieldHandle texfld);
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
  void set_field(FieldHandle tex);
  //////////
  // Change the BrickSize
  bool set_brick_size( int brickSize );
  

  // GROUP: Access
  //////////
  // get min point
  const Point& min() const { return minP_;}
  //////////
  // get max point
  const Point& max() const { return maxP_;}
  /////////
  // the depth of the bontree
  int depth() const { return levels_; }
  /////////
  // the depth of the bontree
  void get_bounds(BBox& b) const { b.extend(minP_); b.extend(maxP_);}
  /////////
  // Get the brick
  int get_brick_size(){ return xmax_; }
  /////////
  // Get field size
  FieldHandle get_field(){ return texfld_; }

  // GROUP: io
  /////////
  // Persistant representation
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  bool CC() const {return isCC_;}
  void getminmax( double& min, double& max) const { min = min_, max = max_;}

private:


  Octree<Brick*>* bontree_;
  FieldHandle texfld_;
  LatVolMeshHandle mesh_;
  int levels_;
  Point minP_;
  Point maxP_;
  double min_, max_;
  int X_, Y_, Z_;
  //  int maxBrick;  // max brick dimension
  int xmax_, ymax_, zmax_;
  double dx_, dy_, dz_;
  bool isCC_;
  double SETVAL(double);
  void set_bounds();
  void compute_tree_depth();
  void build_texture();
  bool set_max_brick_size(int maxBrick);

  template <class T>
    Octree<Brick*>* build_bon_tree(Point min, Point max,
				   int xoff, int yoff, int zoff,
				   int xsize, int ysize, int zsize,
				   int level, T *tex,
				   Octree<Brick*>* parent);
  template <class T>
    void build_child(int i, Point min, Point mid, Point max,
		     int xoff, int yoff, int zoff,
		     int xsize, int ysize, int zsize,
		     int X2, int Y2, int Z2,
		     int level,  T* tex, Octree<Brick*>* node);
  
  template <class T>
    void make_brick_data(int newx, int newy, int newz,
			 int xsize, int ysize, int zsize,
			 int xoff, int yoff, int zoff,
			 T* tex, Array3<unsigned char>*& bd);
  
  template <class T>
    void make_low_res_brick_data(int xmax, int ymax, int zmax,
				 int xsize, int ysize, int zsize,
				 int xoff, int yoff, int zoff,
				 int level, int& padx, int& pady,
				 int& padz, T* tex,
				 Array3<unsigned char>*& bd);
  
};

} // End namespace SCIRun
#endif
