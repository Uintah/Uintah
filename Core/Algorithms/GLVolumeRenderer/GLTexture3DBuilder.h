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

/*
 *  GLTextute3DBuilder:
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2003
 *
 *  Copyright (C) 2003 SCI Group
 */


#ifndef GLTexture3DBuilder_h
#define GLTexture3DBuilder_h

#include <GL/gl.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/GLVolumeRenderer/Brick.h>
#include <Core/GLVolumeRenderer/VolumeUtils.h>
//#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Array3.h>
//#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>
#include <Core/Containers/Octree.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Runnable.h>
#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <sgi_stl_warnings_on.h>

#include <Core/Util/DynamicLoader.h>
#include <Core/GLVolumeRenderer/GLTexture3D.h>

namespace SCIRun {
using std::ostringstream;
class Semaphore;
class ThreadGroup;

// GLTexture3DBuildereAlg

class GLTexture3DBuilderAlg : public DynamicAlgoBase {
public:
  GLTexture3DBuilderAlg();
  virtual ~GLTexture3DBuilderAlg();
  
  void set_caller( GLTexture3D *caller ) { caller_ = caller; }

  virtual Octree<Brick *> *build(Point min, Point max,
				 int xoff, int yoff, int zoff,
				 int xsize, int ysize, int zsize,
				 int level, 
				 Field *tex,
				 Octree<Brick*>* parent,
				 Semaphore* thread_sema, 
				 ThreadGroup* tg) = 0;
  
  virtual void replace_bon_tree_data(Point min, Point max,
				     int xoff, int yoff, int zoff,
				     int xsize, int ysize, int zsize,
				     int level, Field *tex,
				     Octree<Brick*>* parent,
				     Semaphore* thread_sema, ThreadGroup* tg) = 0;

  //! support the dynamically compiled algorithm concept
  static const string& get_h_file_path();
  static CompileInfoHandle get_compile_info(const TypeDescription *td );

  GLTexture3D *caller_;
  double min_, max_;

  void set_minmax( double min, double max ) { min_ = min; max_ = max; }
  double SETVAL(double val)
  {
    double v = (val - min_)*255/(max_ - min_);
    if ( v < 0 ) return 0;
    else if (v > 255) return 255;
    else return v;
  }

  unsigned char SETVALC(double val)
  {
    return (unsigned char)SETVAL(val);
  }
};

// GLTexture3DBuilder<T>

template <class TexField>
class GLTexture3DBuilder : public GLTexture3DBuilderAlg
{
public:
  typedef typename TexField::value_type       value_type;
  
public:
  GLTexture3DBuilder() {}
  virtual ~GLTexture3DBuilder() {}
  
  virtual Octree<Brick *> *build(Point min, Point max,
				 int xoff, int yoff, int zoff,
				 int xsize, int ysize, int zsize,
				 int level, 
				 Field *tex,
				 Octree<Brick*>* parent,
				 Semaphore* thread_sema, 
				 ThreadGroup* tg);
  
  void replace_bon_tree_data(Point min, Point max,
			     int xoff, int yoff, int zoff,
			     int xsize, int ysize, int zsize,
			     int level, Field *tex,
			     Octree<Brick*>* parent,
			     Semaphore* thread_sema, 
			     ThreadGroup* tg);
private:
  void build_child(int i, Point min, Point mid, Point max,
		   int xoff, int yoff, int zoff,
		   int xsize, int ysize, int zsize,
		   int X2, int Y2, int Z2,
		   int level,  TexField *tex, Octree<Brick*>* node,
		   Semaphore* thread_sema, ThreadGroup* tg);
  void fill_child(int i, Point min, Point mid, Point max,
		  int xoff, int yoff, int zoff,
		  int xsize, int ysize, int zsize,
		  int X2, int Y2, int Z2,
		  int level,  TexField *tex, Octree<Brick*>* node,
		  Semaphore* thread_sema, ThreadGroup* tg);
		  
  // friend needed by gcc-2.95.3 compiler
  //template <class T>
#if defined(__GNUC__) && (__GNUC__ == 2) && \
    defined(__GNUC_MINOR__) && (__GNUC_MINOR__ < 96)
  friend 
#endif
  class run_make_brick_data : public Runnable {
  public:
    run_make_brick_data(GLTexture3DBuilderAlg* builder,
		      Semaphore *thread,
		      int newx, int newy, int newz,
		      int xsize, int ysize, int zsize,
		      int xoff, int yoff, int zoff, TexField *tex,
		      Array3<unsigned char>*& bd);
    virtual void run();
  private:
    GLTexture3DBuilderAlg *builder_;
    Semaphore *thread_sema_;
    int newx_, newy_, newz_;
    int xsize_, ysize_, zsize_;
    int xoff_, yoff_, zoff_;
    TexField *tex_;
    Array3<unsigned char>* bd_;
  };

  //  template <class T>
  class run_make_low_res_brick_data : public Runnable {
  public:
    run_make_low_res_brick_data(GLTexture3DBuilderAlg *builder_,
				Semaphore *thread,
				int xmax_, int ymax_, int zmax_,
				int xsize, int ysize, int zsize,
				int xoff, int yoff, int zoff,
				int& padx, int& pady_, int& padz_,
				int level, Octree<Brick*>* node,
				Array3<unsigned char>*& bd);
    virtual void run();
  private:
    GLTexture3DBuilderAlg* builder_;
    Octree<Brick*>* parent_;
    Semaphore *thread_sema_;
    int xmax_, ymax_, zmax_;
    int xsize_, ysize_, zsize_;
    int xoff_, yoff_, zoff_;
    int padx_, pady_, padz_;
    int level_;
    //    TexField tex;
    Array3<unsigned char>* bd_;
  };

};




template <class TexField>
void 
GLTexture3DBuilder<TexField>::replace_bon_tree_data(Point min, Point max,
						    int xoff, int yoff, int zoff,
						    int xsize, int ysize, int zsize,
						    int level, 
						    Field *tex_field, 
						    Octree<Brick*>* parent,
						    Semaphore* thread_sema, 
						    ThreadGroup *)
{
  TexField *tex = dynamic_cast<TexField *>(tex_field);

  Octree<Brick *> *node = parent;

  Brick* brick = (*parent)();
  *(brick->texNameP()) = 0; // needed so that texture resources get deleted
  Array3<unsigned char> *brickData = brick->texture();
  int padx_ = 0,pady_ = 0,padz_ = 0;
  
  if( parent->type() == Octree<Brick *>::LEAF ){
    int newx = xsize, newy = ysize, newz = zsize;
    if (xsize < caller_->xmax_){
      padx_ =  caller_->xmax_ - xsize;
      newx =  caller_->xmax_;
    }
    if (ysize <  caller_->ymax_){
      pady_ =  caller_->ymax_ - ysize;
      newy =  caller_->ymax_;
    }
    if (zsize < caller_->zmax_){
      padz_ =  caller_->zmax_ - zsize;
      newz =  caller_->zmax_;
    }
    
#if 0 //__sgi
    thread_sema->down();
    //Thread *t =   
    scinew Thread(new run_make_brick_data(this,
					  thread_sema, 
					  newx,newy,newz,
					  xsize,ysize,zsize,
					  xoff,yoff,zoff,
					  tex, brickData),
		  "make_brick_data worker", tg);
#else
      run_make_brick_data mbd(this, 
			      thread_sema,
			      newx,newy,newz,
			      xsize,ysize,zsize,
			      xoff,yoff,zoff,
			      tex, brickData);
     mbd.run();
#endif      
  } else {
    double stepx, stepy, stepz;
    stepx = pow(2.0,  caller_->levels_ - level);
    if(  caller_->xmax_ > xsize ) {
      padx_=(int)(( caller_->xmax_ - xsize)*stepx);
    } else {
      if(  caller_->xmax_ * stepx > xsize){
	padx_ = (int)(( caller_->xmax_*stepx - xsize)/stepx);
      }
    }
    stepy = pow(2.0,  caller_->levels_ - level);
    if(  caller_->ymax_ > ysize ) {
      pady_ = (int)(( caller_->ymax_ - ysize)*stepy);
    } else {
      if(  caller_->ymax_ * stepy > ysize){
	pady_ = (int)(( caller_->ymax_*stepy - ysize)/stepy);
      }
    }
    stepz = pow(2.0,  caller_->levels_ - level);
    if(  caller_->zmax_ > zsize ) {
      stepz = 1; padz_ = (int)(( caller_->zmax_ - zsize)*stepz);
    } else {
      if(  caller_->zmax_ * stepz > zsize){
	padz_ = (int)(( caller_->zmax_*stepz - zsize)/stepz);
      }
    }
    string  group_name("thread group ");
    ostringstream osstr;
    osstr << level + 1;
    group_name = group_name + osstr.str();
    ThreadGroup *group = scinew ThreadGroup( group_name.c_str() );
    int sx =  caller_->xmax_, sy =  caller_->ymax_, sz =  caller_->zmax_, tmp;
    tmp =  caller_->xmax_;
    while( tmp < xsize){
      sx = tmp;
      tmp = tmp*2 -1;
    }
    tmp =  caller_->ymax_;
    while( tmp < ysize){
      sy = tmp;
      tmp = tmp*2 -1;
    }
    tmp =  caller_->zmax_;
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
      mid = min + Vector( caller_->dx_* (sx-1),  caller_->dy_* (sy-1),
			  caller_->dz_* (sz-1));
      for(int i = 0; i < 8; i++){
	fill_child(i, min, mid, max, xoff, yoff, zoff,
		   xsize, ysize, zsize, sx, sy, sz,level+1,tex, node, 
		   thread_sema, group);
      }
    } else if( Z2 > Y2 && Z2 > X2 ) {
      mid = min + Vector(diag.x(),
                         diag.y(),
                          caller_->dz_*(sz-1));
      
      fill_child(0, min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, xsize, ysize, sz, level+1, tex, node, 
                          thread_sema, group);
      fill_child(1, min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, xsize, ysize, sz, level+1, tex, node, 
                          thread_sema, group);
    } else  if( Y2 > Z2 && Y2 > X2 ) {
      mid = min + Vector(diag.x(),
                          caller_->dy_*(sy - 1),
                         diag.z());
      fill_child(0, min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, xsize, sy, zsize, level+1, tex, node, 
                          thread_sema, group);
      fill_child(2, min, mid, max, xoff, yoff, zoff,
                 xsize, ysize, zsize, xsize, sy, zsize, level+1, tex, node, 
                          thread_sema, group);
    } else  if( X2 > Z2 && X2 > Y2 ) {
      mid = min + Vector( caller_->dx_*(sx-1),
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
                          caller_->dy_ * (sy - 1),
                          caller_->dz_* (sz - 1));
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
      mid = min + Vector( caller_->dx_*(sx - 1),  caller_->dy_*(sy-1),
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
      mid = min + Vector( caller_->dx_*(sx-1),
                         diag.y(),
                          caller_->dz_*(sz-1));
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
    scinew Thread(new run_make_low_res_brick_data(this, 
						  thread_sema,
						  caller_->xmax_,  caller_->ymax_,  caller_->zmax_,
						  xsize, ysize, zsize,
						  xoff, yoff, zoff, 
						  padx_, pady_, padz_,
						  level, node, brickData),
		  "makeLowResBrickData worker", tg);
#else
    run_make_low_res_brick_data mlrbd(this, 
				      thread_sema,
				      caller_->xmax_,  caller_->ymax_,  caller_->zmax_,
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

template<class TexField>
Octree<Brick *> *
GLTexture3DBuilder<TexField>::build(Point min, Point max,
				    int xoff, int yoff, int zoff,
				    int xsize, int ysize, int zsize,
				    int level, 
				    Field *tex_field,
				    Octree<Brick*>* parent,
				    Semaphore* thread_sema, 
				    ThreadGroup* tg)
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

  TexField *tex = dynamic_cast<TexField*>(tex_field);
  if ( !tex ) {
    cerr << "GLTexture3DBuilder: wrong tex field\n";
    return 0;
  }

  Octree<Brick *> *node;

  if (xoff-caller_->minP_.x() > caller_->X_ 
      || yoff-caller_->minP_.y() > caller_->Y_ 
      || zoff-caller_->minP_.z()> caller_->Z_)
  {
    node = 0;
    return node;
  }

  Brick* brick;
  Array3<unsigned char> *brickData;
  // Check to make sure that we can accommodate the requested texture
  GLint xtex =0 , ytex = 0 , ztex = 0;

  if ( xsize <= caller_->xmax_ ) xtex = 1;
  if ( ysize <= caller_->ymax_ ) ytex = 1;
  if ( zsize <= caller_->zmax_ ) ztex = 1;

  brickData = scinew Array3<unsigned char>();
  int padx_ = 0,pady_ = 0,padz_ = 0;

  if( xtex && ytex && ztex) { // we can accommodate
    int newx = xsize, newy = ysize, newz = zsize;
    if (xsize < caller_->xmax_){
      padx_ = caller_->xmax_ - xsize;
      newx = caller_->xmax_;
    }
    if (ysize < caller_->ymax_){
      pady_ = caller_->ymax_ - ysize;
      newy = caller_->ymax_;
    }
    if (zsize < caller_->zmax_){
      padz_ = caller_->zmax_ - zsize;
      newz = caller_->zmax_;
    }
    brickData->resize( newz, newy, newx);


#ifdef __sgi
    thread_sema->down();
    //Thread *t = 
      scinew Thread(new run_make_brick_data(this, 
					    thread_sema, 
					    newx,newy,newz,
					    xsize,ysize,zsize,
					    xoff,yoff,zoff,
					    tex, brickData),
		    "make_brick_data worker", tg);
#else
      run_make_brick_data mbd(this, 
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

    brickData->resize( caller_->zmax_, caller_->ymax_, caller_->xmax_);

    double stepx, stepy, stepz;

      stepx = pow(2.0, caller_->levels_ - level);
      if( caller_->xmax_ > xsize ) {
	padx_=(int)((caller_->xmax_ - xsize)*stepx);
      } else {
	if( caller_->xmax_ * stepx > xsize){
	  padx_ = (int)((caller_->xmax_*stepx - xsize)/stepx);
	}
      }
      stepy = pow(2.0, caller_->levels_ - level);
      if( caller_->ymax_ > ysize ) {
	pady_ = (int)((caller_->ymax_ - ysize)*stepy);
      } else {
	if( caller_->ymax_ * stepy > ysize){
	  pady_ = (int)((caller_->ymax_*stepy - ysize)/stepy);
	}
      }
      stepz = pow(2.0, caller_->levels_ - level);
      if( caller_->zmax_ > zsize ) {
	stepz = 1; padz_ = (int)((caller_->zmax_ - zsize)*stepz);
      } else {
	if( caller_->zmax_ * stepz > zsize){
	  padz_ = (int)((caller_->zmax_*stepz - zsize)/stepz);
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

    int sx = caller_->xmax_, sy = caller_->ymax_, sz = caller_->zmax_, tmp;
    tmp = caller_->xmax_;
    while( tmp < xsize){
      sx = tmp;
      tmp = tmp*2 -1;
    }
    tmp = caller_->ymax_;
    while( tmp < ysize){
      sy = tmp;
      tmp = tmp*2 -1;
    }
    tmp = caller_->zmax_;
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
    if( Z2 == Y2 && Y2 == X2 ){mid = min + Vector(caller_->dx_* (sx-1), 
						  caller_->dy_* (sy-1),
						  caller_->dz_* (sz-1));
      for(int i = 0; i < 8; i++){
	build_child(i, min, mid, max, xoff, yoff, zoff,
		    xsize, ysize, zsize, sx, sy, sz,level+1,tex, node, 
			  thread_sema, group);
      }
    } else if( Z2 > Y2 && Z2 > X2 ) {
      mid = min + Vector(diag.x(),
			 diag.y(),
			 caller_->dz_*(sz-1));
      
      build_child(0, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, ysize, sz, level+1, tex, node, 
			  thread_sema, group);
      build_child(1, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, ysize, sz, level+1, tex, node, 
			  thread_sema, group);
    } else  if( Y2 > Z2 && Y2 > X2 ) {
      mid = min + Vector(diag.x(),
			 caller_->dy_*(sy - 1),
			 diag.z());
      build_child(0, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, zsize, level+1, tex, node, 
			  thread_sema, group);
      build_child(2, min, mid, max, xoff, yoff, zoff,
		 xsize, ysize, zsize, xsize, sy, zsize, level+1, tex, node, 
			  thread_sema, group);
    } else  if( X2 > Z2 && X2 > Y2 ) {
      mid = min + Vector(caller_->dx_*(sx-1),
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
			 caller_->dy_ * (sy - 1),
			 caller_->dz_* (sz - 1));
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
      mid = min + Vector(caller_->dx_*(sx - 1), caller_->dy_*(sy-1),
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
      mid = min + Vector(caller_->dx_*(sx-1),
			 diag.y(),
			 caller_->dz_*(sz-1));
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
    scinew Thread(new run_make_low_res_brick_data(this, 
						  thread_sema,
						  caller_->xmax_, caller_->ymax_, caller_->zmax_,
						  xsize, ysize, zsize,
						  xoff, yoff, zoff, 
						  padx_, pady_, padz_,
						  level, node, brickData),
		    "makeLowResBrickData worker", tg);
#else
    run_make_low_res_brick_data mlrbd(this, 
				      thread_sema,
				      caller_->xmax_, caller_->ymax_, caller_->zmax_,
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

template <class TexField>
void 
GLTexture3DBuilder<TexField>::build_child(int i, 
					  Point min, Point mid, Point max,
					  int xoff, int yoff, int zoff,
					  int xsize, int ysize, int zsize,
					  int X2, int Y2, int Z2,
					  int level,  TexField *tex, Octree<Brick*>* node,
					  Semaphore* thread_sema, ThreadGroup *tg)
{
  Point pmin, pmax;

  switch( i ) {
  case 0:
    pmin = min;
    pmax = mid;
    node->SetChild(0, build(pmin, pmax, xoff, yoff, zoff,
				     X2, Y2, Z2, level, tex, node, 
				     thread_sema, tg));
    break;
  case 1:
    pmin = min;
    pmax = mid;
    pmin.z(mid.z());
    pmax.z(max.z());
    node->SetChild(1, build(pmin, pmax,
				   xoff, yoff, zoff + Z2 -1,
				   X2, Y2, zsize-Z2+1, level, tex, node, 
				   thread_sema, tg));
    break;
  case 2:
    pmin = min;
    pmax = mid;
    pmin.y(mid.y());
    pmax.y(max.y());
    node->SetChild(2, build(pmin, pmax,
				   xoff, yoff + Y2 - 1, zoff,
				   X2, ysize - Y2 + 1, Z2, level, tex, node, 
				   thread_sema, tg));
    break;
  case 3:
    pmin = mid;
    pmax = max;
    pmin.x(min.x());
    pmax.x(mid.x());
    node->SetChild(3, build(pmin, pmax,
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
    node->SetChild(4, build(pmin, pmax,
				   xoff + X2 - 1, yoff, zoff,
				   xsize - X2 + 1, Y2, Z2, level, tex, node, 
				   thread_sema, tg));
    break;
  case 5:
    pmin = mid;
    pmax = max;
    pmin.y(min.y());
    pmax.y(mid.y());
    node->SetChild(5, build(pmin, pmax,
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
    node->SetChild(6, build(pmin, pmax,
				   xoff + X2 - 1, yoff + Y2 - 1, zoff,
				   xsize - X2 + 1, ysize - Y2 + 1, Z2, level, 
				   tex, node, 
				   thread_sema, tg));
    break;
  case 7:
   pmin = mid;
   pmax = max;
   node->SetChild(7, build(pmin, pmax,  xoff + X2 - 1,
				  yoff + Y2 - 1, zoff +  Z2 - 1,
				  xsize - X2 + 1, ysize - Y2 + 1,
				  zsize - Z2 + 1, level, tex, node, 
				  thread_sema, tg));
   break;
  default:
    break;
  }
}

template <class TexField>
void 
GLTexture3DBuilder<TexField>::fill_child(int i, Point min, Point mid, Point max,
					 int xoff, int yoff, int zoff,
					 int xsize, int ysize, int zsize,
					 int X2, int Y2, int Z2,
					 int level,  TexField *tex, Octree<Brick*>* node,
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



template <class TexField>
GLTexture3DBuilder<TexField>::run_make_brick_data::run_make_brick_data(GLTexture3DBuilderAlg* builder,
								       Semaphore *thread,
								       int newx, int newy, int newz,
								       int xsize, int ysize, int zsize,
								       int xoff, int yoff, int zoff, 
								       TexField *tex,
								       Array3<unsigned char>*& bd) 
  : builder_(builder),
  thread_sema_( thread ),
  newx_(newx), newy_(newy), newz_(newz),
  xsize_(xsize), ysize_(ysize), zsize_(zsize),
  xoff_(xoff), yoff_(yoff), zoff_(zoff),
  tex_(tex), bd_(bd)
{
  // constructor
}

template <class TexField>	
void					
GLTexture3DBuilder<TexField>::run_make_brick_data::run() 
{
  int i,j,k,ii,jj,kk;
  typename TexField::mesh_type *m = tex_->get_typed_mesh().get_rep();

  if( tex_->data_at() == Field::CELL){
    typename TexField::mesh_type mesh(m, xoff_, yoff_, zoff_, 
			       xsize_+1, ysize_+1, zsize_+1);
    typename TexField::mesh_type::Cell::iterator it; mesh.begin(it);
    for(kk = 0, k = zoff_; kk < zsize_; kk++, k++)
      for(jj = 0, j = yoff_; jj < ysize_; jj++, j++)
	for(ii = 0, i = xoff_; ii < xsize_; ii++, i++){
	  (*bd_)(kk,jj,ii) = builder_->SETVALC( tex_->fdata()[*it] );
	  ++it;
	}
  } else {
    typename TexField::mesh_type mesh(m, xoff_, yoff_, zoff_, xsize_, ysize_, zsize_);
    typename TexField::mesh_type::Node::iterator it; mesh.begin(it);
    for(kk = 0, k = zoff_; kk < zsize_; kk++, k++)
      for(jj = 0, j = yoff_; jj < ysize_; jj++, j++)
	for(ii = 0, i = xoff_; ii < xsize_; ii++, i++){
	  (*bd_)(kk,jj,ii) = builder_->SETVALC( tex_->fdata()[*it] );
	  ++it;
	}
  }
#ifdef __sgi
  thread_sema_->up();
#endif  
}

template<class TexField>
GLTexture3DBuilder<TexField>::run_make_low_res_brick_data::run_make_low_res_brick_data(GLTexture3DBuilderAlg* builder,
										       Semaphore *thread,
										       int xmax, int ymax, int zmax,
										       int xsize, int ysize, int zsize,
										       int xoff, int yoff, int zoff,
										       int& padx, int& pady, int& padz,
										       int level, Octree<Brick*>* node,
										       Array3<unsigned char>*& bd) 
  :
  builder_(builder),
  parent_(node), 
  thread_sema_( thread ), 
  xmax_(xmax), 
  ymax_(ymax), 
  zmax_(zmax),
  xsize_(xsize), 
  ysize_(ysize), 
  zsize_(zsize),
  xoff_(xoff), 
  yoff_(yoff), 
  zoff_(zoff),
  padx_(padx), 
  pady_(pady), 
  padz_(padz),
  level_(level), 
  bd_(bd)
{
  // constructor
}


template<class TexField>
void
GLTexture3DBuilder<TexField>::run_make_low_res_brick_data::run() 
{
  using SCIRun::Interpolate;

  int ii,jj,kk;
  Brick *brick = 0;
  Array3<unsigned char>* brickTexture;

//   if( level == 0 ){
//     double  i,j,k;
//     int k1,j1,i1;
//     double dk,dj,di, k00,k01,k10,k11,j00,j01;
//     bool iswitch = false , jswitch = false, kswitch = false;
//     dx = (double)(xsize_-1)/(xmax_-1.0);
//     dy = (double)(ysize_-1)/(ymax_-1.0);
//     dz = (double)(zsize_-1)/(zmax_-1.0);
//     int x,y,z;
//     for( kk = 0, k = 0; kk < zmax_; kk++, k+=dz){
//       if ( dz*kk >= zmax_ )  z = 1; else z = 0;
//       if (!kswitch)
// 	if ( dz*kk >= zmax_ ){ k = zmax_ - dz*kk + 1; kswitch = true; }
//       k1 = ((int)k + 1 >= zmax_)?(int)k:(int)k + 1;
//       if(k1 == (int)k ) { dk = 0; } else {dk = k1 - k;}
//       for( jj = 0, j = 0; jj < ymax_; jj++, j+=dy){
// 	if( dy*jj >= ymax_) y = 2; else y = 0;
// 	if( !jswitch )
// 	  if( dy*jj >= ymax_) { j = ymax_ - dy*jj + 1; jswitch = true; }
// 	j1 = ((int)j + 1 >= ymax_)?(int)j:(int)j + 1 ;
// 	if(j1 == (int)j) {dj = 0;} else { dj = j1 - j;} 
// 	for (ii = 0, i = 0; ii < xmax_; ii++, i+=dx){
// 	  if( dx*ii >= xmax_ ) x = 4; else x = 0;
// 	  if( !iswitch )
// 	    if( dx*ii >= xmax_ ) { i = xmax_ - dz*ii + 1; iswitch = true; }
// 	  i1 = ((int)i + 1 >= xmax_)?(int)i:(int)i + 1 ;
// 	  if( i1 == (int)i){ di = 0;} else {di = i1 - i;}

// 	  brick = (*((*this->parent_)[x+y+z]))();
// 	  if( brick == 0 ){
// 	    (*bd)(kk,jj,ii) = (unsigned char)0;
// 	  } else {
// 	    brickTexture = brick->texture();
// 	    k00 = Interpolate(tex3D_->SETVALC( (*brickTexture)(i,j,k) ),
// 			      tex3D_->SETVALC( (*brickTexture)(i,j,k1)),dk);
// 	    k01 = Interpolate(tex3D_->SETVALC( (*brickTexture)(i1,j,k)),
// 			      tex3D_->SETVALC( (*brickTexture)(i1,j,k1)),dk);
// 	    k10 = Interpolate(tex3D_->SETVALC( (*brickTexture)(i,j1,k)),
// 			      tex3D_->SETVALC( (*brickTexture)(i,j,k1)),dk);
// 	    k11 = Interpolate(tex3D_->SETVALC( (*brickTexture)(i1,j1,k)),
// 			      tex3D_->SETVALC( (*brickTexture)(i1,j1,k1)),dk);
// 	    j00 = Interpolate(k00,k10,dj);
// 	    j01 = Interpolate(k01,k11,dj);
// 	    (*bd_)(kk,jj,ii) = (unsigned char)Interpolate(j00,j01,di);
// 	  }
// 	}
//       }
//     }
// //    thread_sema_->up();
//     return;
//   } else {
    int  i,j,k;
    int x,y,z;
    for( kk = 0, k = 0; kk < zmax_; kk++, k+=2){
      if ( 2*kk >= zmax_ )  z = 1; else z = 0;
      if ( 2*kk == zmax_ ) k = 1;
      for( jj = 0, j = 0; jj < ymax_; jj++, j+=2){
	if( 2*jj >= ymax_) y = 2; else y = 0;
	if( 2*jj == ymax_) j = 1;
	for (ii = 0, i = 0; ii < xmax_; ii++, i+=2){
	  if( 2*ii >= xmax_ ) x = 4; else x = 0;
	  if( 2*ii == xmax_ ) i = 1;

	  const Octree<Brick*>* child = (*this->parent_)[x+y+z];
	  if( child == 0 ){
	    brick = 0;
	  } else {
	    brick = (*child)();
	    brickTexture = brick->texture();
	  }
	  // This code does simple subsampling.  Uncomment the 
	  // center section to perform averaging.
	  if( brick == 0 ){
	    (*bd_)(kk,jj,ii) = (unsigned char)0;
//////////// Uncomment for texel averageing
// 	  } else if((ii > 0 && ii < xmax_ - 1) &&
// 	     (jj > 0 && jj < ymax_ - 1) &&
// 	     (kk > 0 && kk < zmax_ - 1)){
// 	    (*bd_)(kk,jj,ii) = (0.5*(*brickTexture)(k,j,i)           +
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
	    (*bd_)(kk,jj,ii) = (*brickTexture)(k,j,i);
	  }
	}
      }
    }
#ifdef __sgi
    thread_sema_->up();
#endif    
//  }    
}

} // namespace SCIRun

#endif // GLTexture3DBuilder_h
