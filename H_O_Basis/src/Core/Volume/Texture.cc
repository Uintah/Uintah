//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : Texture.cc
//    Author : Milan Ikits
//    Date   : Thu Jul 15 10:58:23 2004

#include <sci_gl.h>

#include <Core/Volume/Texture.h>
#include <Core/Volume/Utils.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>

namespace SCIRun {

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
  : brick_lock_("Texture Access Lock"), nx_(0), ny_(0), nz_(0),
    nc_(0), vmin_(0.0), vmax_(0.0), gmin_(0.0), gmax_(0.0)
{
  nb_[0] = 0;
  nb_[1] = 0;
  bricks_.resize(1);
  bricks_[0].resize(0);
}

Texture::~Texture()
{}

void
Texture::get_sorted_bricks(vector<TextureBrickHandle> &bricks, const Ray& view,
			   int idx)
{
  bricks.clear();
  vector<TextureBrickHandle> &brick_ = bricks_[idx];
  vector<double> dist;
  for (unsigned int i=0; i<brick_.size(); i++)
  {
    Point minp(brick_[i]->bbox().min());
    Point maxp(brick_[i]->bbox().max());
    Vector diag(brick_[i]->bbox().diagonal());
    minp+=diag/1000.;
    maxp-=diag/1000.;
    Point corner[8];
    corner[0] = minp;
    corner[1] = Point(minp.x(), minp.y(), maxp.z());
    corner[2] = Point(minp.x(), maxp.y(), minp.z());
    corner[3] = Point(minp.x(), maxp.y(), maxp.z());
    corner[4] = Point(maxp.x(), minp.y(), minp.z());
    corner[5] = Point(maxp.x(), minp.y(), maxp.z());
    corner[6] = Point(maxp.x(), maxp.y(), minp.z());
    corner[7] = maxp;
    double d;
    for (unsigned int c=0; c<8; c++) {
      double dd=(corner[c]-view.origin()).length();
      if (c==0 || dd<d) d=dd;
    }
    bricks.push_back(brick_[i]);
    dist.push_back(-d);
  }
  Sort(dist, bricks);
}

// clear doesn't really clear everything probably bad programming-kz
void
Texture::clear()
{
  bricks_.clear();

  nx_ = 0;
  ny_ = 0;
  nz_ = 0;
  nc_ = 0;
  vmin_ = 0;
  vmax_ = 0;
  gmin_ = 0;
  gmax_ = 0;
  nb_[0] = 0;
  nb_[1] = 0;
  bricks_.resize(1);
  bricks_[0].resize(0);
}


} // namespace SCIRun
