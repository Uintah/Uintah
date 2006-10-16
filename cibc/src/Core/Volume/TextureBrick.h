//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
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
#include <Core/Geometry/Plane.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Volume/GLinfo.h>
#include <Core/Thread/Mutex.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/NrrdData.h>
#include <vector>

#include <Core/Volume/share.h>

namespace SCIRun {

using std::vector;

#define TEXTURE_MAX_COMPONENTS 2

class SCISHARE TextureBrick
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
  inline int nb(int c)
  {
    ASSERT(c >= 0 && c < TEXTURE_MAX_COMPONENTS);
    return nb_[c];
  }

  inline int mx() { return mx_; }
  inline int my() { return my_; }
  inline int mz() { return mz_; }
  
  inline int ox() { return ox_; }
  inline int oy() { return oy_; }
  inline int oz() { return oz_; }

  virtual int sx() { return 0; }
  virtual int sy() { return 0; }

  virtual GLenum tex_type(int c) = 0;
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

  bool mask_polygons(vector<int> &size,
		     vector<float> &vertex,
		     vector<float> &texcoord,
		     vector<int> &masks,
		     vector<Plane *> &planes);
		     
  enum tb_td_info_e {
    FULL_TD_E,
    TB_NAME_ONLY_E,
    DATA_TD_E
  };

  static const string type_name(int n = -1);
  virtual const TypeDescription* 
  get_type_description(tb_td_info_e td = FULL_TD_E) const = 0;

protected:
  void compute_edge_rays(BBox &bbox, vector<Ray> &edges) const;
  int nx_, ny_, nz_; // axis sizes (pow2)
  int nc_; // number of components (< TEXTURE_MAX_COMPONENTS)
  int nb_[TEXTURE_MAX_COMPONENTS]; // number of bytes per component
  int ox_, oy_, oz_; // offset into volume texture
  int mx_, my_, mz_; // data axis sizes (not necessarily pow2)
  BBox bbox_, tbox_; // bounding box and texcoord box
  vector<Ray> edge_; // bbox edges
  vector<Ray> tex_edge_; // tbox edges
  bool dirty_;

public:
  //! needed for our smart pointers -- LockingHandle<T>
  int ref_cnt;
  Mutex lock;
};

typedef LockingHandle<TextureBrick> TextureBrickHandle;



class SCISHARE NrrdTextureBrick : public TextureBrick
{
public:
  NrrdTextureBrick(NrrdDataHandle n0, NrrdDataHandle n1,
		   int nx, int ny, int nz, int nc, int *nb,
		   int ox, int oy, int oz, int mx, int my, int mz,
		   const BBox& bbox, const BBox& tbox);
  virtual ~NrrdTextureBrick();
  
  virtual GLenum tex_type(int c);
  virtual void *tex_data(int c);

  virtual int sx();
  virtual int sy();

  void set_nrrds(const NrrdDataHandle &n0, const NrrdDataHandle &n1)
  { data_[0] = n0; data_[1] = n1; }

  static bool tex_type_supported(const NrrdDataHandle &n);

  static const string type_name(int n = -1);
  virtual const TypeDescription* 
  get_type_description(tb_td_info_e td = FULL_TD_E) const;

protected:
  NrrdDataHandle data_[TEXTURE_MAX_COMPONENTS];

  static GLenum tex_type_aux(const NrrdDataHandle &n);
  static size_t tex_type_size(GLenum e);
};



template <typename T>
class TextureBrickT : public TextureBrick
{
public:
  TextureBrickT(int nx, int ny, int nz, int nc, int* nb,
		int ox, int oy, int oz,	int mx, int my, int mz,
		const BBox& bbox, const BBox& tbox);
  virtual ~TextureBrickT();
  
  virtual GLenum tex_type(int c) { return GLinfo<T>::type; }
  virtual void* tex_data(int c) { return data_[c]; }
  
  T* data(int c)
  {
    ASSERT(c >= 0 && c < TEXTURE_MAX_COMPONENTS && c < nc());
    return data_[c];
  }

  static const string type_name(int n = -1);
  virtual const TypeDescription* 
  get_type_description(tb_td_info_e td = FULL_TD_E) const;
 
protected:
  T* data_[TEXTURE_MAX_COMPONENTS];
};

template <typename T>
TextureBrickT<T>::TextureBrickT(int nx, int ny, int nz, int nc, int* nb,
				int ox, int oy, int oz,
				int mx, int my, int mz,
				const BBox& bbox, const BBox& tbox)
  : TextureBrick(nx, ny, nz, nc, nb, ox, oy, oz, mx, my, mz, bbox, tbox)
{
  for (int i = 0; i < TEXTURE_MAX_COMPONENTS; i++)
  {
    data_[i] = 0;
    if (i < nc) data_[i] = scinew T[nx_ * ny_ * nz_ * nb_[i]];
  }
}

template <typename T>
TextureBrickT<T>::~TextureBrickT()
{
  for (unsigned int i = 0; i < TEXTURE_MAX_COMPONENTS; i++)
  {
    if (data_[i])
    {
      delete[] data_[i];
      data_[i] = 0;
    }
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
TextureBrickT<T>::get_type_description(tb_td_info_e tdi) const
{
  TypeDescription* td = 0;
  static string name( type_name(0) );
  static string namesp("SCIRun");
  static string path(__FILE__);

  switch (tdi) {
  default:
  case FULL_TD_E:
    {
      static TypeDescription* tdn1 = 0;
      if (tdn1 == 0) {
	const TypeDescription *sub = SCIRun::get_type_description((T*)0);
	TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
	(*subs)[0] = sub;
	tdn1 = scinew TypeDescription(name, subs, path, namesp);
      } 
      td = tdn1;
    }
  case TB_NAME_ONLY_E:
    {
      static TypeDescription* tdn0 = 0;
      if (tdn0 == 0) {
	tdn0 = scinew TypeDescription(name, 0, path, namesp);
      }
      td = tdn0;
    }
  case DATA_TD_E:
    {
      static TypeDescription* tdnn = 0;
      if (tdnn == 0) {
	tdnn = (TypeDescription *) SCIRun::get_type_description((T*)0);
      }
      td = tdnn;
    }
  }
  return td;
} 

} // namespace SCIRun

#endif // Volume_TextureBrick_h
