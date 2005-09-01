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
//    File   : TextureBrick.h
//    Author : Milan Ikits
//    Date   : Wed Jul 14 15:55:55 2004

#ifndef Volume_TextureBrick_h
#define Volume_TextureBrick_h

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Volume/GLinfo.h>
#include <Core/Thread/Mutex.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/NrrdData.h>
#include <vector>

namespace SCIRun {

using std::vector;

class TextureBrick
{
public:
  TextureBrick(int nx, int ny, int nz, int nc, int* nb, int ox, int oy, int oz,
        int mx, int my, int mz, const BBox& bbox, const BBox& tbox);
  virtual ~TextureBrick();

  inline const BBox &bbox() const { return bbox_; }
  //inline BBox tbox() const { return tbox_; }

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

  virtual int sx() { return 0; }
  virtual int sy() { return 0; }

  virtual GLenum tex_type() = 0;
  virtual void* tex_data(int c) = 0;

  inline bool dirty() const { return dirty_; }
  inline void set_dirty(bool b) { dirty_ = b; }
  
  void compute_polygons(const Ray& view, double tmin, double tmax, double dt,
                        vector<float>& vertex, vector<float>& texcoord,
                        vector<int>& size) const;
  void compute_polygons(const Ray& view, double dt,
                        vector<float>& vertex, vector<float>& texcoord,
                        vector<int>& size) const;
  void compute_polygon(const Ray& view, double t,
                       vector<float>& vertex, vector<float>& texcoord,
                       vector<int>& size) const;

  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

protected:
  int nx_, ny_, nz_; // axis sizes (pow2)
  int nc_; // number of components (1 or 2)
  int nb_[2]; // number of bytes per component
  int ox_, oy_, oz_; // offset into volume texture
  int mx_, my_, mz_; // data axis sizes (not necessarily pow2)
  BBox bbox_, tbox_; // bounding box and texcoord box
  Ray edge_[12]; // bbox edges
  Ray tex_edge_[12]; // tbox edges
  bool dirty_;

public:
  //! needed for our smart pointers -- LockingHandle<T>
  int ref_cnt;
  Mutex lock;
};

typedef LockingHandle<TextureBrick> TextureBrickHandle;



class NrrdTextureBrick : public TextureBrick
{
public:
  NrrdTextureBrick(NrrdDataHandle n0, NrrdDataHandle n1,
		   int nx, int ny, int nz, int nc, int *nb,
		   int ox, int oy, int oz, int mx, int my, int mz,
		   const BBox& bbox, const BBox& tbox);
  virtual ~NrrdTextureBrick();
  
  virtual GLenum tex_type();
  virtual void *tex_data(int c);

  virtual int sx();
  virtual int sy();

  void set_nrrds(const NrrdDataHandle &n0, const NrrdDataHandle &n1)
  { data_[0] = n0; data_[1] = n1; }

  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

protected:
  NrrdDataHandle data_[2];
};



template <typename T>
class TextureBrickT : public TextureBrick
{
public:
  TextureBrickT(int nx, int ny, int nz, int nc, int* nb,
		int ox, int oy, int oz,	int mx, int my, int mz,
		const BBox& bbox, const BBox& tbox);
  virtual ~TextureBrickT();
  
  virtual GLenum tex_type() { return GLinfo<T>::type; }
  virtual void* tex_data(int c) { return data_[c]; }
  
  T* data(int c) { return data_[c]; }

  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;
 
protected:
  T* data_[2];
};

template <typename T>
TextureBrickT<T>::TextureBrickT(int nx, int ny, int nz, int nc, int* nb,
				int ox, int oy, int oz,
				int mx, int my, int mz,
				const BBox& bbox, const BBox& tbox)
  : TextureBrick(nx, ny, nz, nc, nb, ox, oy, oz, mx, my, mz, bbox, tbox)
{
  data_[0] = 0;
  data_[1] = 0;
  for (int c=0; c<nc; c++)
  {
    data_[c] = scinew T[nx_*ny_*nz_*nb_[c]];
  }
}

template <typename T>
TextureBrickT<T>::~TextureBrickT()
{
  for (int c=0; c<nc_; c++)
  {
    delete[] data_[c];
  }
}

template <typename T>
const string
TextureBrickT<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "TextureBrickT";
  }
  else
  {
    return find_type_name((T *)0);
  }
}


template <typename T> 
const TypeDescription*
TextureBrickT<T>::get_type_description(int n) const
{
  ASSERT((n >= -1) && n <= 1);

  TypeDescription* td = 0;
  static string name( type_name(0) );
  static string namesp("SCIRun");
  static string path(__FILE__);

  if (n == -1) {
    static TypeDescription* tdn1 = 0;
    if (tdn1 == 0) {
      const TypeDescription *sub = SCIRun::get_type_description((T*)0);
      TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
      (*subs)[0] = sub;
      tdn1 = scinew TypeDescription(name, subs, path, namesp);
    } 
    td = tdn1;
  }
  else if(n == 0) {
    static TypeDescription* tdn0 = 0;
    if (tdn0 == 0) {
      tdn0 = scinew TypeDescription(name, 0, path, namesp);
    }
    td = tdn0;
  }
  else {
    static TypeDescription* tdnn = 0;
    if (tdnn == 0) {
      tdnn = (TypeDescription *) SCIRun::get_type_description((T*)0);
    }
    td = tdnn;
  }
  return td;
} 

} // namespace SCIRun

#endif // Volume_TextureBrick_h
