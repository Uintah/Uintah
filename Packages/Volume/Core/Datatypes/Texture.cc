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

#include <Core/Util/NotFinished.h>
#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Util/DynamicCompilation.h>
#include <Packages/Volume/Core/Datatypes/Texture.h>
#include <Packages/Volume/Core/Datatypes/BrickWindow.h>
#include <Core/Util/DebugStream.h>

#include <math.h>
#include <iostream>
#include <string>
#include <deque>



namespace Volume {
using std::cerr;
using std::endl;
using std::string;
using std::deque;

static DebugStream dbg("Texture", false);

static Persistent* maker()
{
  return scinew Texture;
}

PersistentTypeID Texture::type_id("Texture", "Datatype", maker);

#define Texture_VERSION 3
void Texture::io(Piostream&)
{
    NOT_FINISHED("Texture::io(Piostream&)");
}

Texture::Texture() 
{
}

Texture::Texture(BinaryTree<BrickNode*> *tree,
		 const Point& minP, const Point& maxP,
		 const Transform& trans, double vmin, double vmax,
                 double gmin, double gmax) :
  minP_(minP),  maxP_(maxP),
  transform_(trans),
  vmin_(vmin), vmax_(vmax), gmin_(gmin), gmax_(gmax),
  tree_(tree)
{
}


Texture::~Texture()
{
}



void
Texture::get_sorted_bricks( vector<Brick*>& bricks, const Ray& viewray)
{
  sortBricks( tree_, bricks, viewray);
}

void
Texture::get_dimensions( int& ni, int &nj, int &nk )
{
  BrickWindow *bw = tree_->stored()->brickWindow();
  ni = bw->max_i() - bw->min_i() + 1;
  nj = bw->max_j() - bw->min_j() + 1;
  nk = bw->max_k() - bw->min_k() + 1;
}

void 
Texture::tagBricksForReloading()
{
  tagBricksForReloading( tree_ );
}

void
Texture::tagBricksForReloading(BinaryTree< BrickNode *> *tree)
{
  if( tree->type() == BinaryTree<BrickNode*>::PARENT ){
    tagBricksForReloading( tree->child(0) );
    tagBricksForReloading( tree->child(1) );
  } else {
    Brick *brick = tree->stored()->brick();
    brick->setReload( true );
  }
}

void
Texture::sortBricks(BinaryTree< BrickNode *> *tree,
                    vector<Brick *>& bricks, const Ray& vr)
{
  //char str[80];
//   cerr<<"tree pointer ="<<tree<<"\n";
  if( tree->type() == BinaryTree<BrickNode*>::PARENT ){
    BBox vbox0 = tree->child(0)->stored()->brickWindow()->vbox();
    BBox vbox1 = tree->child(1)->stored()->brickWindow()->vbox();
    //double d0 = (vr.origin() - vbox0.center()).length2();
    //double d1 = (vr.origin() - vbox1.center()).length2();
    int axis = tree->stored()->axis();
//     dbg << "AXIS: " << axis << endl;
//     dbg << vbox0.min() << " -> " << vbox0.max() << endl;
//     dbg << vbox1.min() << " -> " << vbox1.max() << endl;
//     dbg << vr.origin() << endl;
    
    double dp01, dp10;
    int child_first;
    switch (axis) {
    case 1:
      dp01 = vbox0.max().x() - vbox1.min().x();
      dp10 = vbox1.max().x() - vbox0.min().x();
      if (dp01 < std::numeric_limits<float>::epsilon()) {
        child_first = vr.origin().x() > vbox0.max().x() ? 0 : 1;
      } else if (dp10 < std::numeric_limits<float>::epsilon()) {
        child_first = vr.origin().x() > vbox1.max().x() ? 1 : 0;
      }
      break;
    case 2:
      dp01 = vbox0.max().y() - vbox1.min().y();
      dp10 = vbox1.max().y() - vbox0.min().y();
      if (dp01 < std::numeric_limits<float>::epsilon()) {
        child_first = vr.origin().y() > vbox0.max().y() ? 0 : 1;
      } else if (dp10 < std::numeric_limits<float>::epsilon()) {
        child_first = vr.origin().y() > vbox1.max().y() ? 1 : 0;
      }
    case 3:
      dp01 = vbox0.max().z() - vbox1.min().z();
      dp10 = vbox1.max().z() - vbox0.min().z();
      if (dp01 < std::numeric_limits<float>::epsilon()) {
        child_first = vr.origin().z() > vbox0.max().z() ? 0 : 1;
      } else if (dp10 < std::numeric_limits<float>::epsilon()) {
        child_first = vr.origin().z() > vbox1.max().z() ? 1 : 0;
      }
      break;
    default:
      break;
    }
    //dbg << "FIRST: " << child_first << endl;
    sortBricks(tree->child(child_first), bricks, vr);
    sortBricks(tree->child(1-child_first), bricks, vr);
//     cerr<< "\n";
  } else {
    //BBox vbox = tree->stored()->brickWindow()->vbox();
    //double d = (vr.origin() - vbox.center()).length();
    Brick *brick = tree->stored()->brick();
    //dbg << "AXIS: " << tree->stored()->axis() << endl;
//     dbg << "SORT: " << tree->stored()->index() << endl;
    bricks.push_back( brick );
//     sprintf(str,"dist to brick%d is %f, ",
//             tree->stored()->index(), d );
//     cerr<< str ;
  }
}

} // End namespace Volume

