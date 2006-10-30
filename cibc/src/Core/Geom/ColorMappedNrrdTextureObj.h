/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  ColorMappedNrrdTextureObj.h
 *
 *  Written by:
 *   McKay Davis
 *   School of Computing
 *   University of Utah
 *   November, 2005
 *
 *  Copyright (C) 2005 SCI Group
 */

#ifndef SCIRun_Dataflow_Modules_Render_ColorMappedNrrdTextureObj_h
#define SCIRun_Dataflow_Modules_Render_ColorMappedNrrdTextureObj_h


#include <string>
#include <vector>
#include <Core/Datatypes/NrrdData.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/BBox.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Thread/Mutex.h>
#include <Core/Geom/share.h>
namespace SCIRun {

using std::string;


class SCISHARE ColorMappedNrrdTextureObj {
public:
  Mutex lock;
  int ref_cnt;

  ColorMappedNrrdTextureObj(NrrdDataHandle &nrrd_handle,
                            ColorMapHandle &);
  ~ColorMappedNrrdTextureObj();

  void                  set_label(unsigned int);
  void                  set_colormap(ColorMapHandle &cmap);
  void                  set_clut_minmax(float min, float max);
  void                  set_dirty() { dirty_ = true; }
  void                  set_coords(const Point &, 
                                   const Vector &,
                                   const Vector &);  
  void			draw_quad();
  void                  get_bounds(BBox&);
  void                  apply_colormap(int, int, int, int, int border=0);
  void                  set_opacity(float op) { opacity_ = op; } 
private:
  friend class GeomColorMappedNrrdTextureObj;
  bool			bind(int x, int y);

  template <class T> 
  void			apply_colormap_to_raw_data(float *dst,
                                                   T *src,
                                                   int row_width,
                                                   int region_start,
                                                   int region_width,
                                                   int region_height,
                                                   const float *rgba,
                                                   int ncolors,
                                                   float scale, float bias);
  template <class T> 
  void			apply_colormap_to_label_data(float *dst,
                                                     T *src,
                                                     int row_width,
                                                     int region_start,
                                                     int region_width,
                                                     int region_height,
                                                     const float *rgba,
                                                     unsigned char *,
                                                     unsigned char);


  template <class T> 
  void			apply_colormap_to_label_bit(float *dst,
                                                    T *src,
                                                    int row_width,
                                                    int region_start,
                                                    int region_width,
                                                    int region_height,
                                                    const float *rgba,
                                                    unsigned char bit);

  typedef vector<pair<unsigned int, unsigned int> > divisions_t;
  ColorMapHandle        colormap_;
  NrrdDataHandle	nrrd_handle_;
  bool  		dirty_;
  vector<bool>		texture_id_dirty_;
  vector<unsigned int>  texture_id_;
  divisions_t           xdiv_;
  divisions_t           ydiv_;
  float                 clut_min_;
  float                 clut_max_;
  float *               data_;
  unsigned int          label_;
  float                 opacity_;
  Point                 min_;
  Vector                xdir_;
  Vector                ydir_;
};


  typedef LockingHandle<ColorMappedNrrdTextureObj> ColorMappedNrrdTextureObjHandle;



}

  
#endif
