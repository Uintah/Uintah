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
//    File   : Brick.h
//    Author : Milan Ikits
//    Date   : Wed Jul 14 15:55:55 2004

#ifndef Volume_Brick_h
#define Volume_Brick_h

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/BBox.h>
#include <Core/Containers/Array1.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Volume/Core/Util/GLinfo.h>

using SCIRun::Vector;
using SCIRun::Point;
using SCIRun::Ray;
using SCIRun::BBox;
using SCIRun::Array1;

namespace Volume {

class Brick 
{
public:
  Brick(int nx, int ny, int nz, int nc, int* nb, int ox, int oy, int oz,
        int mx, int my, int mz, const BBox& bbox, const BBox& tbox);
  virtual ~Brick();

  // access one of the 8 vertices [0,7]
  inline const Point& operator[] (int i) const { return corner_[i]; }
  inline BBox bbox() const { return bbox_; }
  inline BBox tbox() const { return tbox_; }
  inline Point center() const
  { return corner_[0] + 0.5*(corner_[7] - corner_[0]); }

  inline int nx() { return nx_; }
  inline int ny() { return ny_; }
  inline int nz() { return nz_; }
  inline int nc() { return nc_; }
  inline int nb(int c) { return nb_[c]; }

  inline int mx() { return mx_; }
  inline int my() { return my_; }
  inline int mz() { return mz_; }
  
  inline int ox() { return ox_; }
  inline int oy() { return oy_; }
  inline int oz() { return oz_; }

  virtual GLenum tex_type() = 0;
  virtual void* tex_data(int c) = 0;

  inline bool dirty() const { return dirty_; }
  inline void set_dirty(bool b) { dirty_ = b; }
  
  void compute_polygons(const Ray& view, double tmin, double tmax, double dt,
                        Array1<float>& vertex, Array1<float>& texcoord,
                        Array1<int>& size) const;
  void compute_polygons(const Ray& view, double dt,
                        Array1<float>& vertex, Array1<float>& texcoord,
                        Array1<int>& size) const;
  void compute_polygon(const Ray& view, double t,
                       Array1<float>& vertex, Array1<float>& texcoord,
                       Array1<int>& size) const;

protected:
  int nx_, ny_, nz_; // axis sizes (pow2)
  int nc_; // number of components (1 or 2)
  int nb_[2]; // number of bytes per component
  int ox_, oy_, oz_; // offset into volume texture
  int mx_, my_, mz_; // data axis sizes (not necessarily pow2)
  BBox bbox_, tbox_; // bounding box and texcoord box
  Point corner_[8]; // bbox corners
  Ray edge_[12]; // bbox edges
  Ray tex_edge_[12]; // tbox edges
  bool dirty_;
};

template <typename T>
class BrickT : public Brick
{
public:
  BrickT(int nx, int ny, int nz, int nc, int* nb, int ox, int oy, int oz,
         int mx, int my, int mz, const BBox& bbox, const BBox& tbox, bool alloc);
  ~BrickT();
  
  GLenum tex_type() { return GLinfo<T>::type; }
  void* tex_data(int c) { return data_[c]; }
  T* data(int c) { return data_[c]; }
  
protected:
  T* data_[2];
  bool alloc_;
};

template <typename T>
BrickT<T>::BrickT(int nx, int ny, int nz, int nc, int* nb, int ox, int oy, int oz,
                  int mx, int my, int mz, const BBox& bbox, const BBox& tbox, bool alloc)
  : Brick(nx, ny, nz, nc, nb, ox, oy, oz, mx, my, mz, bbox, tbox)
{
  data_[0] = 0;
  data_[1] = 0;
  if(alloc) {
    for(int c=0; c<nc; c++) {
      data_[c] = scinew T[nx_*ny_*nz_*nb_[c]];
    }
  }
}

template <typename T>
BrickT<T>::~BrickT()
{
  for(int c=0; c<nc_; c++) {
    delete[] data_[c];
  }
}

} // namespace Volume

#endif // Volume_Brick_h
