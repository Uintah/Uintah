//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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
//    File   : NrrdBitmaskOutline.h
//    Author : McKay Davis
//    Date   : Mon Oct 23 18:40:04 2006


#ifndef SCIRun_Dataflow_Modules_Render_NrrdBitmaskOutline_h
#define SCIRun_Dataflow_Modules_Render_NrrdBitmaskOutline_h


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


class SCISHARE NrrdBitmaskOutline {
public:
  Mutex lock;
  int ref_cnt;

  NrrdBitmaskOutline(NrrdDataHandle &nrrd_handle);
  ~NrrdBitmaskOutline();

  void			draw_lines(float width=1, unsigned int mask=0);
  void			draw_quads(float width=1, float sso=1);
  void			draw_glyphs(float width=1, float sso=1);

  void                  set_coords(const Point &, 
                                   const Vector &, 
                                   const Vector &);
  void                  set_colormap(ColorMapHandle &cmap);
  void                  set_dirty(bool dirty = true) { dirty_ = dirty; }
  bool                  get_dirty() { return dirty_; }

  void                  get_bounds(BBox&);

private:

  void                  compute_iso_glyphs(int x1 = -1, int y1 = -1, 
                                           int x2 = -1, int y2 = -1, 
                                           int border = -1);

  void                  resize_iso_glyphs_2D();
  template <class T>
  void                  compute_iso_glyphs_2D(int x1 = -1, int y1 = -1, 
                                              int x2 = -1, int y2 = -1, 
                                              int border = -1);
  

  template <class T> 
  void                  compute_iso_quads(int x1 = -1, int y1 = -1, 
                                          int x2 = -1, int y2 = -1, 
                                          int border = -1);

  template <class T> 
  void                  compute_iso_lines(int x1 = -1, int y1 = -1, 
                                          int x2 = -1, int y2 = -1, 
                                          int border = -1);

  enum cardinal_t {
    SW = 0, S = 1, SE = 2, E = 3, W = 4, NW = 5, N = 6, NE = 7, C = 8, 
    NUM_DIRS = 9
  };
  
  typedef vector<float>         verts_t;
  typedef vector<verts_t>       bit_verts_t;
  NrrdDataHandle                nrrd_handle_;

  vector<vector<vector<pair<size_t, size_t> > > > isoglyphs_;
  vector<vector<vector<vector<bool> > > > isoglyphs2D_;

  bit_verts_t                   line_coords_;
  bit_verts_t                   point_coords_;

  signed char                   bit_priority_[33];
  bit_verts_t                   bit_verts_;
  ColorMapHandle                colormap_;
  bool                          dirty_;
  Point                         min_;
  Vector                        xdir_;
  Vector                        ydir_;
  float                         width_;
  float                         sso_;

};


  typedef LockingHandle<NrrdBitmaskOutline> NrrdBitmaskOutlineHandle;



}

  
#endif
