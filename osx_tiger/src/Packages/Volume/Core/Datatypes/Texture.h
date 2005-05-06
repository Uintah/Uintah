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
//    File   : Texture.h
//    Author : Milan Ikits
//    Date   : Thu Jul 15 01:00:36 2004

#ifndef Volume_Texture_h
#define Volume_Texture_h

#include <Core/Datatypes/Datatype.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Thread/Mutex.h>
#include <Packages/Volume/Core/Datatypes/Brick.h>
#include <Packages/Volume/Core/Util/Utils.h>
#include <vector>

using SCIRun::Datatype;
using SCIRun::Transform;
using SCIRun::Piostream;
using SCIRun::PersistentTypeID;
using SCIRun::LockingHandle;
using SCIRun::Mutex;

namespace Volume {

class Texture : public Datatype
{
public:
  Texture();
  virtual ~Texture();
 
  inline int nx() const { return nx_; }
  inline int ny() const { return ny_; }
  inline int nz() const { return nz_; }

  inline int nc() const { return nc_; }
  inline int nb(int i) const { return nb_[i]; }

  inline void set_size(int nx, int ny, int nz, int nc, int* nb) {
    nx_ = nx; ny_ = ny; nz_ = nz; nc_ = nc;
    for(int c=0; c<nc_; c++) {
      nb_[c] = nb[c];
    }
  }
  
  inline void get_bounds(BBox& b) const {
    b.extend(transform_.project(bbox_.min()));
    b.extend(transform_.project(bbox_.max()));
  }

  inline BBox bbox() const { return bbox_; }
  inline void set_bbox(const BBox& bbox) { bbox_ = bbox; }
  inline Transform transform() const { return transform_; }
  inline void set_transform(Transform tform) { transform_ = tform; }
  
  void get_sorted_bricks(std::vector<Brick*>& bricks, const Ray& view);
  inline std::vector<Brick*>& bricks() { return brick_; }
  
  inline double vmin() const { return vmin_; }
  inline double vmax() const { return vmax_; }
  inline double gmin() const { return gmin_; }
  inline double gmax() const { return gmax_; }
  inline void set_minmax(double vmin, double vmax, double gmin, double gmax) {
    vmin_ = vmin; vmax_ = vmax; gmin_ = gmin; gmax_ = gmax;
  }
  
  inline int card_mem() const { return card_mem_; }
  inline void set_card_mem(int mem) { card_mem_ = mem; }
  
  void lock_bricks() { brick_lock_.lock(); }
  void unlock_bricks() { brick_lock_.unlock(); }
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

protected:
  std::vector<Brick*> brick_;
  Mutex brick_lock_;
  int nx_, ny_, nz_; // data size
  int nc_;
  int nb_[2];
  BBox bbox_; // data bbox
  Transform transform_; // data tform
  double vmin_, vmax_, gmin_, gmax_; //
  int card_mem_;
};

typedef LockingHandle<Texture> TextureHandle;

} // namespace Volume

#endif // Volume_Texture_h
