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

#include <math.h>
#include <iostream>
#include <string>
#include <deque>



namespace Volume {
using std::cerr;
using std::endl;
using std::string;
using std::deque;


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
		 const Transform& trans, double min, double max) :
  minP_(minP),  maxP_(maxP),
  transform_(trans),
  min_(min), max_(max),
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
Texture::sortBricks( BinaryTree< BrickNode *> *tree,
                    vector<Brick *>& bricks, const Ray& vr)
{
  char str[80];
//   cerr<<"tree pointer ="<<tree<<"\n";
  if( tree->type() == BinaryTree<BrickNode*>::PARENT ){
    BBox vbox0 = tree->child(0)->stored()->brickWindow()->vbox();
    BBox vbox1 = tree->child(1)->stored()->brickWindow()->vbox();
    double d0 = (vr.origin() - vbox0.center()).length2();
    double d1 = (vr.origin() - vbox1.center()).length2();
    if( d0 > d1 ){
      sortBricks( tree->child(0), bricks, vr );
      sortBricks( tree->child(1), bricks, vr );
    } else {
      sortBricks( tree->child(1), bricks, vr );
      sortBricks( tree->child(0), bricks, vr );
    }
//     cerr<< "\n";
  } else {
    BBox vbox = tree->stored()->brickWindow()->vbox();
    double d = (vr.origin() - vbox.center()).length();
    Brick *brick = tree->stored()->brick();
    bricks.push_back( brick );
//     sprintf(str,"dist to brick%d is %f, ",
//             tree->stored()->index(), d );
//     cerr<< str ;
  }
}

} // End namespace Volume

