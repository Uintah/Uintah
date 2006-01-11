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
 *  Painter.h
 *
 *  Written by:
 *   McKay Davis
 *   School of Computing
 *   University of Utah
 *   November, 2005
 *
 *  Copyright (C) 2005 SCI Group
 */


#ifndef SCIRun_Dataflow_Modules_Render_Painter_h
#define SCIRun_Dataflow_Modules_Render_Painter_h

#include <sci_comp_warn_fixes.h>
#include <tcl.h>
#include <tk.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <typeinfo>
#include <iostream>
#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/BundlePort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Algorithms/Visualization/RenderField.h>
#include <Core/Bundle/Bundle.h>
#include <Core/Containers/Array3.h>
#include <Core/Datatypes/Field.h> 
#include <Core/Exceptions/GuiException.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/ColorMappedNrrdTextureObj.h>
#include <Core/Geom/FreeTypeTextTexture.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geom/GeomCull.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/TexSquare.h>
#include <Core/Geom/OpenGLContext.h>
#include <Core/Geom/OpenGLViewport.h>
#include <Core/Geom/FreeType.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/UIvar.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Util/Timer.h>
#include <Core/Util/Environment.h>
#include <Core/Volume/CM2Widget.h>

#ifdef _WIN32
#undef min
#undef max
#endif


namespace SCIRun {

class Painter : public Module
{
  struct SliceWindow;
  struct MouseState {
    int                 button_;
    int                 dx_;
    int                 dy_;
    int                 x_;
    int                 y_;
    int                 X_;
    int                 Y_;
    int                 pick_x_;
    int                 pick_y_;
    int                 state_;
    Point               position_;
    SliceWindow *       window_;
    enum {
      SHIFT_E           = 1,
      CAPS_LOCK_E       = 2,
      CONTROL_E         = 4,
      ALT_E             = 8,
      M1_E              = 16,
      M2_E              = 32,
      M3_E              = 64,
      M4_E              = 128,
      BUTTON_1_E	= 256,
      BUTTON_2_E        = 512,
      BUTTON_3_E        = 1024
    };
  };

  struct WindowLayout;
  struct NrrdSlice;
  typedef vector<NrrdSlice *>		NrrdSlices;
  typedef vector<NrrdSlices>		NrrdVolumeSlices;

  // needs to go before all SliceWindow& usage
  struct SliceWindow { 
    SliceWindow(Painter &painter, GuiContext *ctx);
    void                setup_gl_view();
    void		next_slice();
    void		prev_slice();
    void		zoom_in();
    void		zoom_out();
    Point		world_to_screen(const Point &);
    Point		screen_to_world(unsigned int x, unsigned int y);
    Vector		x_dir();
    Vector		y_dir();
    int                 x_axis();
    int                 y_axis();
    void                render_text();
    void		render_orientation_text();
    void		render_grid();
    void                draw_line(const Point &, const Point &);

    Painter &           painter_;
    string		name_;
    WindowLayout *	layout_;
    OpenGLViewport *	viewport_;
    NrrdSlices		slices_;

    Point               center_;
    Vector              normal_;

    UIint		slice_num_;
    UIint		axis_;
    UIdouble		zoom_;
    UIint		slab_min_;
    UIint		slab_max_;
      
    bool		redraw_;
    bool                autoview_;
    UIint		mode_;
    UIint		show_guidelines_;
    int			cursor_pixmap_;

    GLdouble		gl_modelview_matrix_[16];
    GLdouble		gl_projection_matrix_[16];
    GLint		gl_viewport_[4];
  };
  friend struct SliceWindow;

  typedef vector<SliceWindow *>	SliceWindows;
    

  class PainterTool {
  public:
    PainterTool(Painter *painter, const string &name) : 
      painter_(painter), name_(name) {}
    virtual ~PainterTool() {};
    const string &      get_name()      { return name_; };

    virtual string *    mouse_button_press(MouseState &) { return 0; }
    virtual string *    mouse_button_release(MouseState &) { return 0; }
    virtual string *    mouse_motion(MouseState &) { return 0; }

    virtual string *    draw(SliceWindow &) { return 0; }
    virtual string *    draw_mouse_cursor(MouseState &) { return 0; }
  protected:
    Painter *           painter_;
    string              name_;
  };

  class CLUTLevelsTool : public PainterTool {
  public:
    CLUTLevelsTool(Painter *painter);
    string *            mouse_button_press(MouseState &);
    string *            mouse_button_release(MouseState &);
    string *            mouse_motion(MouseState &);
  private:
    double              scale_;
    double              ww_;
    double              wl_;
  };


  class ZoomTool : public PainterTool {
  public:
    ZoomTool(Painter *painter);
    string *            mouse_button_press(MouseState &);
    string *            mouse_button_release(MouseState &);
    string *            mouse_motion(MouseState &);
  private:
    double              zoom_;
    SliceWindow *       window_;
  };

  class AutoviewTool : public PainterTool {
  public:
    AutoviewTool(Painter *painter);
    string *            mouse_button_press(MouseState &);
    string *            mouse_button_release(MouseState &);
  };

  class ProbeTool : public PainterTool {
  public:
    ProbeTool(Painter *painter);
    string *            mouse_button_press(MouseState &);
    string *            mouse_button_release(MouseState &);
    string *            mouse_motion(MouseState &);
  };


  class PanTool : public PainterTool {
  public:
    PanTool(Painter *painter);
    string *            mouse_button_press(MouseState &);
    string *            mouse_button_release(MouseState &);
    string *            mouse_motion(MouseState &);
  private:
    Point               center_;
    SliceWindow *       window_;
  };




  class CropTool : public PainterTool {
  public:
    CropTool(Painter *painter);
    ~CropTool();
    string *            mouse_button_press(MouseState &);
    string *            mouse_button_release(MouseState &);
    string *            mouse_motion(MouseState &);
    string *            draw(SliceWindow &window);
  private:
    typedef vector<BBox> BBoxes;
    void                compute_crop_pick_boxes(SliceWindow &);
    void                update_bbox_from_gui();
    void                update_bbox_to_gui();
    int                 get_pick_from_mouse(MouseState &mouse);
    void                set_window_cursor(SliceWindow &window, int cursor);
    pair<Vector,Vector> get_crop_vectors(SliceWindow &window, int pick);
    
    int                 pick_;
    BBox	        bbox_;
    BBox		draw_bbox_;
    BBoxes		pick_boxes_;
    
    int			crop_min_x_;
    int			crop_min_y_;
    int			crop_min_z_;
    
    int			crop_max_x_;
    int			crop_max_y_;
    int			crop_max_z_;
    
    double              x_;
    double              y_;
  };


  class BrushTool : public PainterTool {
  public:
    BrushTool(Painter *painter);
    ~BrushTool();
    string *            mouse_button_press(MouseState &);
    string *            mouse_button_release(MouseState &);
    string *            mouse_motion(MouseState &);
    string *            draw(SliceWindow &window);
    string *            draw_mouse(MouseState &);
  private:
    double              value_;
    double              radius_;
  };

  class FloodfillTool : public PainterTool {
  public:
    FloodfillTool(Painter *painter);
    ~FloodfillTool();
    string *            mouse_button_press(MouseState &);
    string *            mouse_button_release(MouseState &);
    string *            mouse_motion(MouseState &);
    string *            draw(SliceWindow &window);
    string *            draw_mouse(MouseState &);
  private:
    double              value_;
    double              min_;
    double              max_;
    Point               start_pos_;
  };

  class PixelPaintTool : public PainterTool {
  public:
    PixelPaintTool(Painter *painter);
    ~PixelPaintTool();
    string *            mouse_button_press(MouseState &);
    string *            mouse_button_release(MouseState &);
    string *            mouse_motion(MouseState &);
  private:
    double              value_;
  };

  class NrrdVolume;
  class ITKThresholdTool : public PainterTool {
  public:
    ITKThresholdTool(Painter *painter, bool test);
    ~ITKThresholdTool();
    string *            mouse_button_press(MouseState &);
    string *            mouse_button_release(MouseState &);
    string *            mouse_motion(MouseState &);
  private:
    NrrdVolume *        volume_;
    double              value_;
  };

  class NrrdVolume;
  class StatisticsTool : public PainterTool {
  public:
    StatisticsTool(Painter *painter);
    ~StatisticsTool();
    //    string *            mouse_button_press(MouseState &);
    string *            mouse_button_release(MouseState &);
    string *            mouse_motion(MouseState &);
    string *            draw(SliceWindow &window);
  private:
    void                recompute();
    double              standard_deviation_;
    double              mean_;
    vector<double>      values_;
  };
    


  friend class PainterTool;

  enum DisplayMode_e {
    normal_e,
    slab_e,
    mip_e,
    num_display_modes_e
  };

  class NrrdVolume { 
  public:
    NrrdVolume		(GuiContext *ctx, 
                         const string &name,
                         NrrdDataHandle &);
    NrrdVolume          (NrrdVolume *copy, const string &name, bool clear = 0);
    void                set_nrrd(NrrdDataHandle &);
    NrrdDataHandle      get_nrrd();
    Point               index_to_world(const vector<int> &index);
    vector<int>         world_to_index(const Point &p);
    vector<double>      vector_to_index(const Vector &v);
    DenseMatrix         build_index_to_world_matrix();
    bool                index_valid(const vector<int> &index);
    template<class T>
    void                get_value(const vector<int> &index, T &value);
    template<class T>
    void                set_value(const vector<int> &index, T value);

    Point               center(int axis = -1, int slice = -1);
    Point               min(int axis = -1, int slice = -1);
    Point               max(int axis = -1, int slice = -1);

    Vector              scale();
    double              scale(int axis);

    vector<int>         max_index();
    int                 max_index(int axis);

    bool                inside_p(const Point &p);
    NrrdDataHandle	nrrd_;
    GuiContext *        gui_context_;
    GuiString           name_;
    UIdouble		opacity_;
    UIdouble            clut_min_;
    UIdouble            clut_max_;
    Semaphore           semaphore_;
    float               data_min_;
    float               data_max_;
    GuiInt              colormap_;
    vector<int>         stub_axes_;
  };

  struct WindowLayout;

  struct NrrdSlice {
    NrrdSlice(Painter *, NrrdVolume *, SliceWindow *);
    void                bind();
    void                draw();
    void	        set_coords();
    Painter *           painter_;
    NrrdVolume *	volume_;
    SliceWindow	*	window_;

    bool		nrrd_dirty_;
    bool		tex_dirty_;
    bool		geom_dirty_;

    Point               pos_;
    Vector              xdir_;
    Vector              ydir_;

    ColorMappedNrrdTextureObj *    texture_;

    //    Mutex		lock_;
    //    Thread *	owner_;
    //    int		lock_count_;
    void		do_lock();
    void		do_unlock();
  };

  struct WindowLayout {
    WindowLayout	(GuiContext *ctx);
    TkOpenGLContext *	opengl_;
    SliceWindows	windows_;
    string		name_;
  };
  typedef map<string, WindowLayout *>	WindowLayouts;

  typedef vector<NrrdVolume *>		NrrdVolumes;
  typedef map<string, NrrdVolume *>	NrrdVolumeMap;


  class RealDrawer : public Runnable {
    Painter *           module_;
    TimeThrottle	throttle_;
  public:
    bool		dead_;
    RealDrawer(Painter* module) : module_(module), throttle_(), dead_(0) {};
    virtual ~RealDrawer();
    virtual void run();
  };
  
  WindowLayouts		layouts_;
  NrrdVolumes		volumes_;
  NrrdVolumeMap         volume_map_;
  NrrdVolume *          current_volume_;

  typedef map<string, ColorMapHandle> colormap_map_t;

  colormap_map_t	colormaps_;
  vector<string>        colormap_names_;
  PainterTool *         tool_;
  MouseState            mouse_;

  int			cur_slice_[3];
  int			slab_width_[3];

  UIint			anatomical_coordinates_;
  UIint			show_text_;
  UIdouble		font_r_;
  UIdouble		font_g_;
  UIdouble		font_b_;
  UIdouble		font_a_;

  //! Ports
  BundleOPort *         bundle_oport_;
  vector<BundleHandle>  bundles_;

  FreeTypeLibrary *	freetype_lib_;
  typedef		map<string, FreeTypeFace *> FontMap;
  FontMap		fonts_;
  UIdouble		font_size_;
  
  RealDrawer *		runner_;
  Thread *		runner_thread_;
  int                   filter_;

  // Methods for drawing to the GL window
  void			redraw_all();
  int			redraw_window(SliceWindow &);
  void			draw_guide_lines(SliceWindow &, float, float, float);
  void			draw_slice_lines(SliceWindow &);

  
  // Methods to render TrueType text labels
  void			initialize_fonts();
  void			delete_all_fonts();
  void			set_font_sizes(double size);



  int			extract_window_slices(SliceWindow &);

  // Methods for navigating around the slices
  void			set_axis(SliceWindow &, unsigned int axis);

  // Methods for cursor/coordinate projection and its inverse
  int			x_axis(SliceWindow &);
  int			y_axis(SliceWindow &);

  // Methods called by tcl_command
  void                  update_mouse_state(GuiArgs &args, bool reset = false);
  void			handle_gui_mouse_motion(GuiArgs &args);
  void			handle_gui_mouse_button_press(GuiArgs &args);
  void			handle_gui_mouse_button_release(GuiArgs &args);
  void			handle_gui_keypress(GuiArgs &args);
  void			handle_gui_mouse_enter(GuiArgs &args);
  void			handle_gui_mouse_leave(GuiArgs &args);


  typedef int (SCIRun::Painter::* SliceWindowFunc)(SliceWindow &);
  typedef int (SCIRun::Painter::* WindowLayoutFunc)(WindowLayout &);
  typedef int (SCIRun::Painter::* NrrdSliceFunc)(NrrdSlice &);
  template <class T> int			for_each(T);
  int			for_each(SliceWindow &, NrrdSliceFunc);
  int			for_each(WindowLayout &, NrrdSliceFunc);
  int			for_each(WindowLayout &, SliceWindowFunc);

  int			render_window(SliceWindow &);
  int			swap_window(SliceWindow &);
  int			autoview(SliceWindow &);
  int			mark_autoview(SliceWindow &);

  int			rebind_slice(NrrdSlice &);
  int			set_slice_nrrd_dirty(NrrdSlice &);

  int                   set_probe(SliceWindow &window);

  int                   create_volume(NrrdVolumes *copies = 0);
  NrrdVolume *          create_volume(string name, int nrrdType);
  ColorMapHandle        get_colormap(int id);
  void                  send_data();
  bool                  receive_filter_data();

public:
  Painter(GuiContext* ctx);
  virtual ~Painter();
  virtual void		execute();
  virtual void		tcl_command(GuiArgs& args, void*);
  void			real_draw_all();

  static bool           static_callback(void *);
  bool                  callback();

  virtual void          set_context(Network *);
  double		fps_;
  WindowLayout *	current_layout_;
  int			executing_;
};





template<class T>
static void
nrrd_get_value(const Nrrd *nrrd, 
               const vector<int> &index, 
               T &value)
{
  ASSERT(int(index.size()) == nrrd->dim);
  int position = index[0];
  int mult_factor = nrrd->axis[0].size;
  for (int a = 1; a < nrrd->dim; ++a) {
    position += index[a] * mult_factor;
    mult_factor *= nrrd->axis[a].size;
  }

  switch (nrrd->type) {
  case nrrdTypeChar: {
    char *slicedata = (char *)nrrd->data;
    value = (T)slicedata[position];
  } break;
  case nrrdTypeUChar: {
    unsigned char *slicedata = (unsigned char *)nrrd->data;
    value = (T)slicedata[position];
  } break;
  case nrrdTypeShort: {
    short *slicedata = (short *)nrrd->data;
    value = (T)slicedata[position];
  } break;
  case nrrdTypeUShort: {
    unsigned short *slicedata = (unsigned short *)nrrd->data;
    value = (T)slicedata[position];
  } break;
  case nrrdTypeInt: {
    int *slicedata = (int *)nrrd->data;
    value = (T)slicedata[position];
  } break;
  case nrrdTypeUInt: {
    unsigned int *slicedata = (unsigned int *)nrrd->data;
    value = (T)slicedata[position];
  } break;
  case nrrdTypeLLong: {
    signed long long *slicedata = (signed long long *)nrrd->data;
    value = (T)slicedata[position];
  } break;
  case nrrdTypeULLong: {
    unsigned long long *slicedata = (unsigned long long *)nrrd->data;
    value = (T)slicedata[position];
  } break;
  case nrrdTypeFloat: {
    float *slicedata = (float *)nrrd->data;
    value = (T)slicedata[position];
  } break;
  case nrrdTypeDouble: {
    double *slicedata = (double *)nrrd->data;
    value = (T)slicedata[position];
  } break;
  default: {
    throw "Unsupported data type: "+to_string(nrrd->type);
  } break;
  }
}



template <class T>
static void
nrrd_set_value(Nrrd *nrrd, 
               const vector<int> &index, 
               T val)
{
  ASSERT(int(index.size()) == nrrd->dim);
  int position = index[0];
  int mult_factor = nrrd->axis[0].size;
  for (int a = 1; a < nrrd->dim; ++a) {
    position += index[a] * mult_factor;
    mult_factor *= nrrd->axis[a].size;
  }

  switch (nrrd->type) {
  case nrrdTypeChar: {
    char *slicedata = (char *)nrrd->data;
    slicedata[position] = (char)val;
  } break;
  case nrrdTypeUChar: {
    unsigned char *slicedata = (unsigned char *)nrrd->data;
    slicedata[position] = (unsigned char)val;
    } break;
  case nrrdTypeShort: {
    short *slicedata = (short *)nrrd->data;
    slicedata[position] = (short)val;
    } break;
  case nrrdTypeUShort: {
    unsigned short *slicedata = (unsigned short *)nrrd->data;
    slicedata[position] = (unsigned short)val;
    } break;
  case nrrdTypeInt: {
    int *slicedata = (int *)nrrd->data;
    slicedata[position] = (int)val;
    } break;
  case nrrdTypeUInt: {
    unsigned int *slicedata = (unsigned int *)nrrd->data;
    slicedata[position] = (unsigned int)val;
    } break;
  case nrrdTypeLLong: {
    signed long long *slicedata = (signed long long *)nrrd->data;
    slicedata[position] = (signed long long)val;
    } break;
  case nrrdTypeULLong: {
    unsigned long long *slicedata = (unsigned long long *)nrrd->data;
    slicedata[position] = (unsigned long long)val;
    } break;
  case nrrdTypeFloat: {
    float *slicedata = (float *)nrrd->data;
    slicedata[position] = (float)val;
    } break;
  case nrrdTypeDouble: {
    double *slicedata = (double *)nrrd->data;
    slicedata[position] = (double)val;
    } break;
  default: { 
    throw "Unsupported data type: "+to_string(nrrd->type);
    } break;
  }
}



template<class T>
void
Painter::NrrdVolume::get_value(const vector<int> &index, T &value) {
  ASSERT(index_valid(index));
  nrrd_get_value(nrrd_->nrrd, index, value);
}


template <class T>
void
Painter::NrrdVolume::set_value(const vector<int> &index, T value) {
  ASSERT(index_valid(index));
  nrrd_set_value(nrrd_->nrrd, index, value);
}


}

#endif
