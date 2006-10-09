/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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


class SCISHARE BBoxSet {
public:
  BBoxSet();
  BBoxSet(BBox &bbox);
  BBoxSet(BBoxSet &bbox);
  ~BBoxSet();

  typedef vector<BBox> BBoxes;

  void          add(BBox &);
  void          add(BBoxSet &);

  void          sub(BBox &);
  void          sub(BBoxSet &);

  void          reset();

  void          set(BBox &);
  void          set(BBoxSet &);

  BBox          get();
  vector<BBox>  get_boxes();


private:
  BBoxes         boxes_;
};
 

class SCISHARE ColorMappedNrrdTextureObj {
public:
  Mutex &lock;
  int ref_cnt;

  ColorMappedNrrdTextureObj(NrrdDataHandle &nrrd_handle, 
                            int axis, 
                            int min_slice, int max_slice,
                            int time = 0);

  ~ColorMappedNrrdTextureObj();

  void                  set_colormap(ColorMapHandle &cmap);
  void                  set_clut_minmax(float min, float max);
  void                  set_dirty() { nrrd_dirty_ = true; }
  void			draw_quad(Point *min=0, Vector *xdir=0, Vector *ydir=0);
  bool                  dirty_p() { return nrrd_dirty_; }
  void                  get_bounds(BBox&);

  void                  apply_colormap(int, int, int, int, int border=0);
  ColorMapHandle        colormap_;
  NrrdDataHandle	nrrd_handle_;

private:
  void                  create_data();
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
  bool  		nrrd_dirty_;
  vector<bool>		dirty_;
  BBoxSet               dirty_region_;
  vector<unsigned int>  texture_id_;
  vector<pair<unsigned int, unsigned int> >  xdiv_;
  vector<pair<unsigned int, unsigned int> >  ydiv_;
  float                 clut_min_;
  float                 clut_max_;
  float *               data_;
  bool                  own_data_;

  Point                 min_;
  Vector                xdir_;
  Vector                ydir_;

};


  typedef LockingHandle<ColorMappedNrrdTextureObj> ColorMappedNrrdTextureObjHandle;



}

  
#endif
