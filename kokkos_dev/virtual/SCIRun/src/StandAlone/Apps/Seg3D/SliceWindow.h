//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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
//    File   : SliceWindow.h
//    Author : McKay Davis
//    Date   : Fri Oct 13 16:05:01 2006

#ifndef SEG3D_SliceWindow
#define SEG3D_SliceWindow

#include <StandAlone/Apps/Seg3D/VolumeSlice.h>
#include <vector>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/NrrdData.h>
#include <Core/Skinner/Parent.h>
#include <Core/Skinner/Variables.h>
#include <Core/Geom/IndexedGroup.h>
#include <Core/Geom/OpenGLViewport.h>


using std::vector;


namespace SCIRun {

class SliceWindow;
typedef vector<SliceWindow *>	SliceWindows;

class SliceWindow : public Skinner::Parent {
public:
  SliceWindow(Skinner::Variables *, Painter *painter);
  virtual ~SliceWindow() {}
    
  virtual int           get_signal_id(const string &signalname) const;
  CatcherFunction_t     Autoview;
  CatcherFunction_t     zoom_in;
  CatcherFunction_t     zoom_out;
  CatcherFunction_t     redraw;
  CatcherFunction_t     do_PointerEvent;
  CatcherFunction_t     process_event;


  void                  setup_gl_view();
  void                  move_slice(int amount);

  Point                 world_to_screen(const Point &);
  Point                 screen_to_world(unsigned int x, unsigned int y);
  Vector		x_dir();
  Vector		y_dir();
  int                   x_axis();
  int                   y_axis();

  void                  render_text();
  void                  render_orientation_text();
  void                  render_grid();
  void                  render_guide_lines(Point);
  void                  render_slice_lines(SliceWindows &);
  void                  render_slices();

  void                  mark_redraw();
  void                  set_probe();
  void                  extract_slices();
  void                  autoview(NrrdVolumeHandle &, double offset=10.0);
  void                  set_axis(unsigned int);
  GeomIndexedGroup*     get_geom_group();

  Painter *             painter_;
  string		name_;
  VolumeSlices_t	slices_;
  bool                  recompute_slices_;

  Point                 center_;
  Vector                normal_;

  int                   axis_;
  double		zoom_;
  int                   slab_min_;
  int                   slab_max_;
      
  int                   show_guidelines_;
  unsigned int          pdown_;

  GLdouble		gl_modelview_matrix_[16];
  GLdouble		gl_projection_matrix_[16];
  GLint                 gl_viewport_[4];

  Skinner::Var<Skinner::Color>    color_;
  Skinner::Var<bool>    show_grid_;
  Skinner::Var<bool>    show_slices_;
  Skinner::Var<string>  groupname_;
  GeomHandle            geom_switch_;
  GeomIndexedGroup *    geom_group_;

};




}
#endif
