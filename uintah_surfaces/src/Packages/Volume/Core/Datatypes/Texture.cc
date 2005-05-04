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

#include <Packages/Volume/Core/Datatypes/Texture.h>
#include <Packages/Volume/Core/Util/Utils.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>

using namespace SCIRun;

namespace Volume {

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
}

Texture::~Texture()
{}

void
Texture::get_sorted_bricks(vector<Brick*>& bricks, const Ray& view)
{
  bricks.resize(0);
  vector<double> dist;
  for(unsigned int i=0; i<brick_.size(); i++) {
    bricks.push_back(brick_[i]);
    dist.push_back(-(brick_[i]->center()-view.origin()).length());
  }
  Sort(dist, bricks);
}

} // namespace Volume
