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

//#define use_alg

#include <Core/Datatypes/Datatype.h>

#include <Core/Containers/Array3.h>
#include <Core/Containers/BinaryTree.h>
#include <Core/Containers/LockingHandle.h>

#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/Transform.h>
#include <Core/Util/ProgressReporter.h>

#include <Packages/Volume/Core/Datatypes/Brick.h>
#include <Packages/Volume/Core/Datatypes/BrickNode.h>
#include <Packages/Volume/Core/Util/Utils.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sstream>
#include <sgi_stl_warnings_off.h>

namespace Volume {

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
using std::ostringstream;
using SCIRun::BinaryTree;
using SCIRun::Transform;
using SCIRun::Point;
using SCIRun::Ray;



class Texture : public Datatype {

public:
  // GROUP: Constructors:
  //////////
  // Constructor
  Texture(BinaryTree<BrickNode*> *tree,
	  const Point& minP, const Point& maxP,
	  const Transform& trans, double min, double max);
  //////////
  // Constructor
  Texture();
  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~Texture();
 
  // GROUP: Modify
  //////////  

  // GROUP: Access
  //////////
  // get min point -- IN FIELD INDEX SPACE
  const Point& min() const { return minP_;}
  //////////
  // get max point -- IN FIELD INDEX SPACE
  const Point& max() const { return maxP_;}
  /////////
  // get the texture tree root pointer
  BinaryTree<BrickNode*>* getTree(){ return tree_;}
  /////////
  // the bounding box -- IN WORLD SPACE
  void get_bounds(BBox& b) const 
  {
    b.extend(transform_.project(minP_)); b.extend(transform_.project(maxP_));
  }
  /////////
  // get the over all texture dimensions
  void get_dimensions( int& ni, int &nj, int &nk );
  /////////
  // Get the bricks
  void get_sorted_bricks( vector<Brick*>& bricks, const Ray& viewray);
  /////////
  // return the min and max data value
  void get_min_max( double& min, double& max){ min = min_; max = max_;}
  /////////
  // return the field_transform
  Transform  get_field_transform(){ return transform_;}
  /////////
  // Persistant representation
  virtual void io(Piostream&);
  static PersistentTypeID type_id;


protected:

  Transform transform_;

  Point minP_, maxP_;
  double min_, max_;
  
  BinaryTree<BrickNode*> *tree_;  
// #ifdef use_alg
//   ProgressReporter my_reporter_;
//   template<class Reporter> void build_texture( Reporter *);
//   template<class Reporter> void replace_texture( Reporter *);
//   void build_texture();
//   void replace_texture();
// #endif

  void sortBricks( BinaryTree< BrickNode *> *tree,
		   vector<Brick *>& bricks, const Ray& vr);
};

typedef LockingHandle<Texture> TextureHandle;


} // End namespace Volume
#endif

