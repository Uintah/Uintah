
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

#define use_alg

#include <GL/gl.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/GLVolumeRenderer/Brick.h>
#include <Core/GLVolumeRenderer/VolumeUtils.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Array3.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>
#include <Core/Containers/Octree.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Runnable.h>
#include <Core/Util/ProgressReporter.h>
#include <iostream>
#include <sstream>
#include <deque>

namespace SCIRun {
using std::deque;
using std::ostringstream;
class Semaphore;
class ThreadGroup;


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
  GLTexture3D(FieldHandle texfld, double &min, double &max, bool use_minmax,
	      int bricksize = 128);
  //////////
  // Constructor
  GLTexture3D();
  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~GLTexture3D();
 
  // GROUP: Modify
  //////////  
  // Set a new scalarField
  virtual void set_field(FieldHandle tex);
  //////////
  // Change the BrickSize
  virtual bool set_brick_size( int brickSize );
  //////////
  // Just reuse the allocated Bricks
  bool replace_data(FieldHandle texfld, 
		    double &min, double &max,
		    bool use_minmax);

  /////////
  // sometimes we want several textures to use the same slices
  // these bounds are set in index space! may or may not be good
  void set_slice_bounds(const Point& min, const Point& max){
      sminP_ = min; smaxP_ = max; slice_bounds_set = true; }
  /////////
  // Hack needed for multipipe rendering--should not be used otherwise
    void set_bounds( const Point& min, const Point& max){
      minP_ = min; maxP_ = max; }

  // GROUP: Access
  //////////
  // get min point -- IN FIELD INDEX SPACE
  const Point& min() const { return minP_;}
  //////////
  // get max point -- IN FIELD INDEX SPACE
  const Point& max() const { return maxP_;}
  /////////
  // the depth of the bontree
  int depth() const { return levels_; }
  /////////
  // the bounding box -- IN WORLD SPACE
  void get_bounds(BBox& b) const {
    b.extend(transform_.project(minP_)); b.extend(transform_.project(maxP_)); }
    // b.extend(texfld_->mesh()->get_bounding_box().min()); b.extend(texfld_->mesh()->get_bounding_box().max()); } //b.extend(minP_); b.extend(maxP_);}
  /////////
  // the bounding box for slices in WORLD SPACE
  void get_slice_bounds(BBox & b) const {
    b.extend(transform_.project(sminP_)); 
    b.extend(transform_.project(smaxP_));}

  bool has_slice_bounds(){return slice_bounds_set;}
  /////////
  // Get the brick
  int get_brick_size(){ return xmax_; }
  /////////
  // Get field size
  FieldHandle get_field(){ return texfld_; }
  Transform get_field_transform() const { return transform_; }

  // GROUP: io
  /////////
  // Persistant representation
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  bool CC() const {return isCC_;}
  void getminmax( double& min, double& max) const { min = min_, max = max_;}
  virtual bool get_dimensions( int& nx, int& ny, int& nz );
  // yarden: fixme
public: // protected:
  static int max_workers;
  ThreadGroup *tg;
  Octree<Brick*>* bontree_;
  FieldHandle texfld_;
  int levels_;
  Point minP_, maxP_;
  Point sminP_, smaxP_;
  bool slice_bounds_set;
  double min_, max_;
  int X_, Y_, Z_;
  //  int maxBrick;  // max brick dimension
  int xmax_, ymax_, zmax_;
  double dx_, dy_, dz_;
  bool isCC_;
  bool reuse_bricks;

  bool init(double& min, double &max, bool use_minmax);
  void set_bounds();
  void compute_tree_depth();

#ifdef use_alg
  ProgressReporter my_reporter_;
  template<class Reporter> void build_texture( Reporter *);
  template<class Reporter> void replace_texture( Reporter *);
  void build_texture() { build_texture( &my_reporter_); }
  void replace_texture() { replace_texture( &my_reporter_); }
#else
  void build_texture();
  void replace_texture();

  template <class T>
    Octree<Brick*>* build_bon_tree(Point min, Point max,
				   int xoff, int yoff, int zoff,
				   int xsize, int ysize, int zsize,
				   int level, T *tex,
				   Octree<Brick*>* parent,
				  Semaphore* thread_sema, ThreadGroup* tg);
  template <class T>
    void replace_bon_tree_data(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int level, T *tex,
			       Octree<Brick*>* parent,
			       Semaphore* thread_sema, ThreadGroup* tg);


private:

  double SETVAL(double);
  unsigned char SETVALC(double);

#endif

  Transform transform_;

  bool set_max_brick_size(int maxBrick);
#ifndef use_alg
  template <class T>
    void build_child(int i, Point min, Point mid, Point max,
		     int xoff, int yoff, int zoff,
		     int xsize, int ysize, int zsize,
		     int X2, int Y2, int Z2,
		     int level,  T* tex, Octree<Brick*>* node,
		     Semaphore* thread_sema, ThreadGroup* tg);
  template <class T>
    void fill_child(int i, Point min, Point mid, Point max,
		     int xoff, int yoff, int zoff,
		     int xsize, int ysize, int zsize,
		     int X2, int Y2, int Z2,
		     int level,  T* tex, Octree<Brick*>* node,
		     Semaphore* thread_sema, ThreadGroup* tg);
  
  // friend needed by gcc-2.95.3 compiler
  template <class T>
#if defined(__GNUC__) && (__GNUC__ == 2) && \
    defined(__GNUC_MINOR__) && (__GNUC_MINOR__ < 96)
  friend 
#endif
  class run_make_brick_data : public Runnable {
  public:
    run_make_brick_data(GLTexture3D* tex3D,
		      Semaphore *thread,
		      int newx, int newy, int newz,
		      int xsize, int ysize, int zsize,
		      int xoff, int yoff, int zoff, T *tex,
		      Array3<unsigned char>*& bd);
    virtual void run();
  private:
    GLTexture3D *tex3D_;
    Semaphore *thread_sema_;
    int newx_, newy_, newz_;
    int xsize_, ysize_, zsize_;
    int xoff_, yoff_, zoff_;
    T* tex_;
    Array3<unsigned char>* bd_;
  };

  //  template <class T>
  class run_make_low_res_brick_data : public Runnable {
  public:
    run_make_low_res_brick_data(GLTexture3D* tex3D,
				Semaphore *thread,
				int xmax_, int ymax_, int zmax_,
				int xsize, int ysize, int zsize,
				int xoff, int yoff, int zoff,
				int& padx, int& pady_, int& padz_,
				int level, Octree<Brick*>* node,
				Array3<unsigned char>*& bd);
    virtual void run();
    // yarden: fixme
    //private:
  public:
    GLTexture3D* tex3D_;
    Octree<Brick*>* parent_;
    Semaphore *thread_sema_;
    int xmax_, ymax_, zmax_;
    int xsize_, ysize_, zsize_;
    int xoff_, yoff_, zoff_;
    int padx_, pady_, padz_;
    int level_;
    //    T* tex;
    Array3<unsigned char>* bd_;
  };
#endif
};


#ifndef use_alg
template <class T>
void 
GLTexture3D::replace_bon_tree_data(Point min, Point max,
				   int xoff, int yoff, int zoff,
				   int xsize, int ysize, int zsize,
				   int level, T *tex, Octree<Brick*>* parent,
				   Semaphore* thread_sema, ThreadGroup *)
{
  Octree<Brick *> *node = parent;

  Brick* brick = (*parent)();
  *(brick->texNameP()) = 0; // needed so that texture resources get deleted
  Array3<unsigned char> *brickData = brick->texture();
  int padx_ = 0,pady_ = 0,padz_ = 0;
  
  if( parent->type() == Octree<Brick *>::LEAF ){
    int newx = xsize, newy = ysize, newz = zsize;
    if (xsize < xmax_){
      padx_ = xmax_ - xsize;
      newx = xmax_;
    }
    if (ysize < ymax_){
      pady_ = ymax_ - ysize;
      newy = ymax_;
    }
    if (zsize < zmax_){
      padz_ = zmax_ - zsize;
      newz = zmax_;
    }
    
#if 0 //__sgi
    thread_sema->down();
    //Thread *t = 
      scinew Thread(new GLTexture3D::run_make_brick_data<T>(this, 
                                                            thread_sema, 
                                                            newx,newy,newz,
                                                            xsize,ysize,zsize,
                                                            xoff,yoff,zoff,
                                                            tex, brickData),
                    "make_brick_data worker", tg);
#else
     GLTexture3D::run_make_brick_data<T> mbd(this, 
                                          thread_sema,
                                          newx,newy,newz,
                                          xsize,ysize,zsize,
                                          xoff,yoff,zoff,
                                          tex, brickData);
     mbd.run();
#endif      
  } else {
    double stepx, stepy, stepz;
    stepx = pow(2.0, levels_ - level);
    if( xmax_ > xsize ) {
      padx_=(int)((xmax_ - xsize)*stepx);
    } else {
      if( xmax_ * stepx > xsize){
	padx_ = (int)((xmax_*stepx - xsize)/stepx);
      }
    }
    stepy = pow(2.0, levels_ - level);
    if( ymax_ > ysize ) {
      pady_ = (int)((ymax_ - ysize)*stepy);
    } else {
      if( ymax_ * stepy > ysize){
	pady_ = (int)((ymax_*stepy - ysize)/stepy);
      }
    }
    stepz = pow(2.0, levels_ - level);
    if( zmax_ > zsize ) {
      stepz = 1; padz_ = (int)((zmax_ - zsize)*stepz);
    } else {
      if( zmax_ * stepz > zsize){
	padz_ = (int)((zmax_*stepz - zsize)/stepz);
      }
    }
    string  group_name("thread group ");
    ostringstream osstr;
    osstr << level + 1;
    group_name = group_name + osstr.str();
    ThreadGroup *group = scinew ThreadGroup( group_name.c_str() );
    int sx = xmax_, sy = ymax_, sz = zmax_, tmp;
    tmp = xmax_;
    while( tmp < xsize){
      sx = tmp;
      tmp = tmp*2 -1;
    }
    tmp = ymax_;
    while( tmp < ysize){
      sy = tmp;
      tmp = tmp*2 -1;
    }
    tmp = zmax_;
    while( tmp < zsize){
      sz = tmp;
      tmp = tmp*2 -1;
    }   

    int X2, Y2, Z2;
    X2 = largestPowerOf2( xsize -1);
    Y2 = largestPowerOf2( ysize -1);
    Z2 = largestPowerOf2( zsize -1);


      
    Vector diag = max - min;
    Point mid;
    if( Z2 == Y2 && Y2 == X2 ){
      mid = min + Vector(dx_* (sx-1), dy_* (sy-1),
			 dz_* (sz-1));
      for(int i = 0; i < 8; i++){
	fill_child(i, min, mid, max, xoff, yoff, zoff,
		   xsize, ysize, zsize, sx, sy, sz,level+1,tex, node, 
		   thread_sema, group);
      }
    } else if( Z2 > Y2 && Z2 > X2 ) {
      mid = min + Vector(diag.x(),
                         diag.y(),
                         dz_*(sz-1));
      
      fill_child(0, min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, xsize, ysize, sz, level+1, tex, node, 
                          thread_sema, group);
      fill_child(1, min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, xsize, ysize, sz, level+1, tex, node, 
                          thread_sema, group);
    } else  if( Y2 > Z2 && Y2 > X2 ) {
      mid = min + Vector(diag.x(),
                         dy_*(sy - 1),
                         diag.z());
      fill_child(0, min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, xsize, sy, zsize, level+1, tex, node, 
                          thread_sema, group);
      fill_child(2, min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, xsize, sy, zsize, level+1, tex, node, 
                          thread_sema, group);
    } else  if( X2 > Z2 && X2 > Y2 ) {
      mid = min + Vector(dx_*(sx-1),
                         diag.y(),
                         diag.z());
      fill_child(0, min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, sx, ysize, zsize, level+1, tex, node, 
                          thread_sema, group);
      fill_child(4,min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, sx, ysize, zsize, level+1, tex, node, 
                          thread_sema, group);
    } else if( Z2 == Y2 ){
      mid = min + Vector(diag.x(),
                         dy_ * (sy - 1),
                         dz_* (sz - 1));
      fill_child(0,min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, xsize, sy, sz, level+1, tex, node, 
                          thread_sema, group);
      fill_child(1,min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, xsize, sy, sz, level+1, tex, node, 
                          thread_sema, group);
      fill_child(2,min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, xsize, sy, sz, level+1, tex, node, 
                          thread_sema, group);
      fill_child(3,min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, xsize, sy, sz, level+1, tex, node, 
                          thread_sema, group);
    } else if( X2 == Y2 ){
      mid = min + Vector(dx_*(sx - 1), dy_*(sy-1),
                         diag.z());
      fill_child(0,min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, sx, sy, zsize, level+1, tex, node, 
                          thread_sema, group);
      fill_child(2,min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, sx, sy, zsize, level+1, tex, node, 
                          thread_sema, group);
      fill_child(4,min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, sx, sy, zsize, level+1, tex, node, 
                          thread_sema, group);
      fill_child(6,min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, sx, sy, zsize, level+1, tex, node, 
                          thread_sema, group);
    } else if( Z2 == X2 ){
      mid = min + Vector(dx_*(sx-1),
                         diag.y(),
                         dz_*(sz-1));
      fill_child(0,min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, sx, ysize, sz, level+1, tex, node, 
                          thread_sema, group);
      fill_child(1,min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, sx, ysize, sz, level+1, tex, node, 
                          thread_sema, group);
      fill_child(4,min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, sx, ysize, sz, level+1, tex, node, 
                          thread_sema, group);
      fill_child(5,min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, sx, ysize, sz, level+1, tex, node, 
                          thread_sema, group);
    }

#if 0 // __sgi
    group->join();
    //group->stop();
      
    thread_sema->down();
    //    Thread *t =
    scinew Thread(new GLTexture3D::run_make_low_res_brick_data(this, 
                                                    thread_sema,
                                                    xmax_, ymax_, zmax_,
                                                    xsize, ysize, zsize,
                                                    xoff, yoff, zoff, 
                                                    padx_, pady_, padz_,
                                                    level, node, brickData),
                    "makeLowResBrickData worker", tg);
#else
      GLTexture3D::run_make_low_res_brick_data mlrbd(this, 
                                                   thread_sema,
                                                   xmax_, ymax_, zmax_,
                                                   xsize, ysize, zsize,
                                                   xoff, yoff, zoff, 
                                                   padx_, pady_, padz_,
                                                   level, node, brickData);

     mlrbd.run();
#endif
    if(group->numActive(false) != 0){
      cerr<<"Active Threads in thread group\n";
    }
    delete group;
  }
}

      

  


template <class T>
Octree<Brick*>*
GLTexture3D::build_bon_tree(Point min, Point max,
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

  if (xoff-minP_.x() > X_ || yoff-minP_.y() > Y_ || zoff-minP_.z()> Z_){
    node = 0;
    return node;
  }

  Brick* brick;
  Array3<unsigned char> *brickData;
  // Check to make sure that we can accommodate the requested texture
  GLint xtex =0 , ytex = 0 , ztex = 0;

  if ( xsize <= xmax_ ) xtex = 1;
  if ( ysize <= ymax_ ) ytex = 1;
  if ( zsize <= zmax_ ) ztex = 1;

  brickData = scinew Array3<unsigned char>();
  int padx_ = 0,pady_ = 0,padz_ = 0;

  if( xtex && ytex && ztex) { // we can accommodate
    int newx = xsize, newy = ysize, newz = zsize;
    if (xsize < xmax_){
      padx_ = xmax_ - xsize;
      newx = xmax_;
    }
    if (ysize < ymax_){
      pady_ = ymax_ - ysize;
      newy = ymax_;
    }
    if (zsize < zmax_){
      padz_ = zmax_ - zsize;
      newz = zmax_;
    }
    brickData->resize( newz, newy, newx);


#ifdef __sgi
    thread_sema->down();
    //Thread *t = 
      scinew Thread(new GLTexture3D::run_make_brick_data<T>(this, 
							    thread_sema, 
							    newx,newy,newz,
							    xsize,ysize,zsize,
							    xoff,yoff,zoff,
							    tex, brickData),
		    "make_brick_data worker", tg);
#else
     GLTexture3D::run_make_brick_data<T> mbd(this, 
 					  thread_sema,
 					  newx,newy,newz,
 					  xsize,ysize,zsize,
 					  xoff,yoff,zoff,
 					  tex, brickData);
     mbd.run();
#endif      
    
    brick = scinew Brick(min, max, padx_, pady_, padz_, level, brickData);

    node = scinew Octree<Brick*>(brick, Octree<Brick *>::LEAF, parent );

  } else { // we must subdivide

    brickData->resize( zmax_, ymax_, xmax_);

    double stepx, stepy, stepz;

      stepx = pow(2.0, levels_ - level);
      if( xmax_ > xsize ) {
	padx_=(int)((xmax_ - xsize)*stepx);
      } else {
	if( xmax_ * stepx > xsize){
	  padx_ = (int)((xmax_*stepx - xsize)/stepx);
	}
      }
      stepy = pow(2.0, levels_ - level);
      if( ymax_ > ysize ) {
	pady_ = (int)((ymax_ - ysize)*stepy);
      } else {
	if( ymax_ * stepy > ysize){
	  pady_ = (int)((ymax_*stepy - ysize)/stepy);
	}
      }
      stepz = pow(2.0, levels_ - level);
      if( zmax_ > zsize ) {
	stepz = 1; padz_ = (int)((zmax_ - zsize)*stepz);
      } else {
	if( zmax_ * stepz > zsize){
	  padz_ = (int)((zmax_*stepz - zsize)/stepz);
	}
      }

    string  group_name("thread group ");
    ostringstream osstr;
    osstr << level + 1;
    group_name = group_name + osstr.str();
    ThreadGroup *group = scinew ThreadGroup( group_name.c_str() );
    
    brick = scinew Brick(min, max, padx_, pady_, padz_, level, brickData);
    
    node = scinew Octree<Brick*>(brick, Octree<Brick *>::PARENT,
				    parent);

    int sx = xmax_, sy = ymax_, sz = zmax_, tmp;
    tmp = xmax_;
    while( tmp < xsize){
      sx = tmp;
      tmp = tmp*2 -1;
    }
    tmp = ymax_;
    while( tmp < ysize){
      sy = tmp;
      tmp = tmp*2 -1;
    }
    tmp = zmax_;
    while( tmp < zsize){
      sz = tmp;
      tmp = tmp*2 -1;
    }   
 


    int X2, Y2, Z2;
    X2 = largestPowerOf2( xsize -1);
    Y2 = largestPowerOf2( ysize -1);
    Z2 = largestPowerOf2( zsize -1);


      
    Vector diag = max - min;
    Point mid;
    if( Z2 == Y2 && Y2 == X2 ){mid = min + Vector(dx_* (sx-1), dy_* (sy-1),
						  dz_* (sz-1));
      for(int i = 0; i < 8; i++){
	build_child(i, min, mid, max, xoff, yoff, zoff,
		    xsize, ysize, zsize, sx, sy, sz,level+1,tex, node, 
			  thread_sema, group);
      }
    } else if( Z2 > Y2 && Z2 > X2 ) {
      mid = min + Vector(diag.x(),
			 diag.y(),
			 dz_*(sz-1));
      
      build_child(0, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, ysize, sz, level+1, tex, node, 
			  thread_sema, group);
      build_child(1, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, ysize, sz, level+1, tex, node, 
			  thread_sema, group);
    } else  if( Y2 > Z2 && Y2 > X2 ) {
      mid = min + Vector(diag.x(),
			 dy_*(sy - 1),
			 diag.z());
      build_child(0, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, zsize, level+1, tex, node, 
			  thread_sema, group);
      build_child(2, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, zsize, level+1, tex, node, 
			  thread_sema, group);
    } else  if( X2 > Z2 && X2 > Y2 ) {
      mid = min + Vector(dx_*(sx-1),
			 diag.y(),
			 diag.z());
      build_child(0, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, zsize, level+1, tex, node, 
			  thread_sema, group);
      build_child(4,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, zsize, level+1, tex, node, 
			  thread_sema, group);
    } else if( Z2 == Y2 ){
      mid = min + Vector(diag.x(),
			 dy_ * (sy - 1),
			 dz_* (sz - 1));
      build_child(0,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level+1, tex, node, 
			  thread_sema, group);
      build_child(1,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level+1, tex, node, 
			  thread_sema, group);
      build_child(2,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level+1, tex, node, 
			  thread_sema, group);
      build_child(3,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, sz, level+1, tex, node, 
			  thread_sema, group);
    } else if( X2 == Y2 ){
      mid = min + Vector(dx_*(sx - 1), dy_*(sy-1),
			 diag.z());
      build_child(0,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level+1, tex, node, 
			  thread_sema, group);
      build_child(2,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level+1, tex, node, 
			  thread_sema, group);
      build_child(4,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level+1, tex, node, 
			  thread_sema, group);
      build_child(6,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, sy, zsize, level+1, tex, node, 
			  thread_sema, group);
    } else if( Z2 == X2 ){
      mid = min + Vector(dx_*(sx-1),
			 diag.y(),
			 dz_*(sz-1));
      build_child(0,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level+1, tex, node, 
			  thread_sema, group);
      build_child(1,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level+1, tex, node, 
			  thread_sema, group);
      build_child(4,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level+1, tex, node, 
			  thread_sema, group);
      build_child(5,min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, sx, ysize, sz, level+1, tex, node, 
			  thread_sema, group);
    }

#ifdef __sgi
    group->join();
      
    thread_sema->down();
    scinew Thread(new GLTexture3D::run_make_low_res_brick_data(this, 
						    thread_sema,
						    xmax_, ymax_, zmax_,
						    xsize, ysize, zsize,
						    xoff, yoff, zoff, 
						    padx_, pady_, padz_,
						    level, node, brickData),
		    "makeLowResBrickData worker", tg);
#else
      GLTexture3D::run_make_low_res_brick_data mlrbd(this, 
 						   thread_sema,
 						   xmax_, ymax_, zmax_,
 						   xsize, ysize, zsize,
 						   xoff, yoff, zoff, 
 						   padx_, pady_, padz_,
 						   level, node, brickData);

     mlrbd.run();
#endif
    if(group->numActive(false) != 0){
      cerr<<"Active Threads in thread group\n";
    }
    delete group;
  }
  return node;
}

template <class T>
void GLTexture3D::build_child(int i, Point min, Point mid, Point max,
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
    node->SetChild(0, build_bon_tree(pmin, pmax, xoff, yoff, zoff,
				   X2, Y2, Z2, level, tex, node, 
			   thread_sema, tg));
    break;
  case 1:
    pmin = min;
    pmax = mid;
    pmin.z(mid.z());
    pmax.z(max.z());
    node->SetChild(1, build_bon_tree(pmin, pmax,
				   xoff, yoff, zoff + Z2 -1,
				   X2, Y2, zsize-Z2+1, level, tex, node, 
				   thread_sema, tg));
    break;
  case 2:
    pmin = min;
    pmax = mid;
    pmin.y(mid.y());
    pmax.y(max.y());
    node->SetChild(2, build_bon_tree(pmin, pmax,
				   xoff, yoff + Y2 - 1, zoff,
				   X2, ysize - Y2 + 1, Z2, level, tex, node, 
				   thread_sema, tg));
    break;
  case 3:
    pmin = mid;
    pmax = max;
    pmin.x(min.x());
    pmax.x(mid.x());
    node->SetChild(3, build_bon_tree(pmin, pmax,
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
    node->SetChild(4, build_bon_tree(pmin, pmax,
				   xoff + X2 - 1, yoff, zoff,
				   xsize - X2 + 1, Y2, Z2, level, tex, node, 
				   thread_sema, tg));
    break;
  case 5:
    pmin = mid;
    pmax = max;
    pmin.y(min.y());
    pmax.y(mid.y());
    node->SetChild(5, build_bon_tree(pmin, pmax,
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
    node->SetChild(6, build_bon_tree(pmin, pmax,
				   xoff + X2 - 1, yoff + Y2 - 1, zoff,
				   xsize - X2 + 1, ysize - Y2 + 1, Z2, level, 
				   tex, node, 
				   thread_sema, tg));
    break;
  case 7:
   pmin = mid;
   pmax = max;
   node->SetChild(7, build_bon_tree(pmin, pmax,  xoff + X2 - 1,
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
void GLTexture3D::fill_child(int i, Point min, Point mid, Point max,
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
    replace_bon_tree_data(pmin, pmax, xoff, yoff, zoff,
			  X2, Y2, Z2, level, tex, (*node)[i], 
			  thread_sema, tg);
    break;
  case 1:
    pmin = min;
    pmax = mid;
    pmin.z(mid.z());
    pmax.z(max.z());
    replace_bon_tree_data(pmin, pmax,
			  xoff, yoff, zoff + Z2 -1,
			  X2, Y2, zsize-Z2+1, level, tex, (*node)[i], 
			  thread_sema, tg);
    break;
  case 2:
    pmin = min;
    pmax = mid;
    pmin.y(mid.y());
    pmax.y(max.y());
    replace_bon_tree_data(pmin, pmax,
			  xoff, yoff + Y2 - 1, zoff,
			  X2, ysize - Y2 + 1, Z2, level, tex, (*node)[i], 
			  thread_sema, tg);
    break;
  case 3:
    pmin = mid;
    pmax = max;
    pmin.x(min.x());
    pmax.x(mid.x());
    replace_bon_tree_data(pmin, pmax,
			  xoff, yoff + Y2 - 1 , zoff + Z2 - 1,
			  X2, ysize - Y2 + 1, zsize - Z2 + 1, level, 
			  tex, (*node)[i], 
			  thread_sema, tg);
    break;
  case 4:
    pmin = min;
    pmax = mid;
    pmin.x(mid.x());
    pmax.x(max.x());
    replace_bon_tree_data(pmin, pmax,
			  xoff + X2 - 1, yoff, zoff,
			  xsize - X2 + 1, Y2, Z2, level, tex, (*node)[i], 
			  thread_sema, tg);
    break;
  case 5:
    pmin = mid;
    pmax = max;
    pmin.y(min.y());
    pmax.y(mid.y());
    replace_bon_tree_data(pmin, pmax,
			  xoff + X2 - 1, yoff, zoff +  Z2 - 1,
			  xsize - X2 + 1, Y2, zsize - Z2 + 1, level, 
			  tex, (*node)[i], 
			  thread_sema, tg);
    break;
  case 6:
    pmin = mid;
    pmax = max;
    pmin.z(min.z());
    pmax.z(mid.z());
    replace_bon_tree_data(pmin, pmax,
			  xoff + X2 - 1, yoff + Y2 - 1, zoff,
			  xsize - X2 + 1, ysize - Y2 + 1, Z2, level, 
			  tex, (*node)[i], 
			  thread_sema, tg);
    break;
  case 7:
    pmin = mid;
    pmax = max;
    replace_bon_tree_data(pmin, pmax,  xoff + X2 - 1,
			  yoff + Y2 - 1, zoff +  Z2 - 1,
			  xsize - X2 + 1, ysize - Y2 + 1,
			  zsize - Z2 + 1, level, tex, (*node)[i], 
			  thread_sema, tg);
    break;
  default:
    break;
  }
}

template <class T>
GLTexture3D::run_make_brick_data<T>::run_make_brick_data(
				 GLTexture3D* tex3D,
				 Semaphore *thread,
				 int newx, int newy, int newz,
				 int xsize, int ysize, int zsize,
				 int xoff, int yoff, int zoff, T *tex,
				 Array3<unsigned char>*& bd) :
  tex3D_(tex3D),
  thread_sema_( thread ),
  newx_(newx), newy_(newy), newz_(newz),
  xsize_(xsize), ysize_(ysize), zsize_(zsize),
  xoff_(xoff), yoff_(yoff), zoff_(zoff),
  tex_(tex), bd_(bd)
{
  // constructor
}

template <class T>	
void					
GLTexture3D::run_make_brick_data<T>::run() 
{
  int i,j,k,ii,jj,kk;
  typename T::mesh_type *m = tex_->get_typed_mesh().get_rep();

  if( tex_->data_at() == Field::CELL){
    typename T::mesh_type mesh(m, xoff_, yoff_, zoff_, 
			       xsize_+1, ysize_+1, zsize_+1);
    typename T::mesh_type::Cell::iterator it; mesh.begin(it);
    for(kk = 0, k = zoff_; kk < zsize_; kk++, k++)
      for(jj = 0, j = yoff_; jj < ysize_; jj++, j++)
	for(ii = 0, i = xoff_; ii < xsize_; ii++, i++){
	  (*bd_)(kk,jj,ii) = tex3D_->SETVALC( tex_->fdata()[*it] );
	  ++it;
	}
  } else {
    typename T::mesh_type mesh(m, xoff_, yoff_, zoff_, xsize_, ysize_, zsize_);
    typename T::mesh_type::Node::iterator it; mesh.begin(it);
    for(kk = 0, k = zoff_; kk < zsize_; kk++, k++)
      for(jj = 0, j = yoff_; jj < ysize_; jj++, j++)
	for(ii = 0, i = xoff_; ii < xsize_; ii++, i++){
	  (*bd_)(kk,jj,ii) = tex3D_->SETVALC( tex_->fdata()[*it] );
	  ++it;
	}
  }
#ifdef __sgi
  thread_sema_->up();
#endif  
}

#endif
} // End namespace SCIRun
#endif

