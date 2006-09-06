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
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Ports/GeometryPort.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Core/Algorithms/Visualization/RenderField.h>
#include <Core/Bundle/Bundle.h>
#include <Core/Containers/Array3.h>
#include <Core/Datatypes/Field.h> 
#include <Core/Exceptions/GuiException.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/ColorMappedNrrdTextureObj.h>
#include <Core/Geom/TextRenderer.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geom/GeomCull.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/TexSquare.h>
#include <Dataflow/TkExtensions/TkOpenGLContext.h>
#include <Core/Geom/OpenGLViewport.h>
#include <Core/Geom/FreeType.h>
#include <Core/Geometry/Plane.h>
#include <Dataflow/GuiInterface/TCLTask.h>
#include <Dataflow/GuiInterface/UIvar.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
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
  struct Event {
    void                update_state(GuiArgs &args, Painter &painter);
    bool                button(unsigned int);
    bool                shift();
    bool                control();
    bool                alt();
    int                 x_;
    int                 y_;
    int                 X_;
    int                 Y_;
    int                 button_;
    int                 state_;
    Point               position_;
    SliceWindow *       window_;
    SliceWindow *       last_window_;
    set<string>         keys_;
    string              key_;
    int                 type_;

    enum {
      SHIFT_E           = 1 << 0,
      CAPS_LOCK_E       = 1 << 1,
      CONTROL_E         = 1 << 2,
      ALT_E             = 1 << 3,
      M1_E              = 1 << 4,
      M2_E              = 1 << 5,
      M3_E              = 1 << 6,
      M4_E              = 1 << 7,
      BUTTON_1_E	= 1 << 8,
      BUTTON_2_E        = 1 << 9,
      BUTTON_3_E        = 1 << 10,
      BUTTON_4_E        = 1 << 11,
      BUTTON_5_E        = 1 << 12
    };
    
    enum {
      MOUSE_MOTION_E    = 1 << 0,
      BUTTON_PRESS_E    = 1 << 1,
      BUTTON_RELEASE_E  = 1 << 2,
      KEY_PRESS_E       = 1 << 3,
      KEY_RELEASE_E     = 1 << 4,
      FOCUS_IN_E        = 1 << 5,
      FOCUS_OUT_E       = 1 << 6
    };
  };
  friend struct Event;
  
  class NrrdVolume;
  struct WindowLayout;

  struct NrrdSlice {
    NrrdSlice(Painter *, NrrdVolume *, Point &p, Vector &normal);
    void                bind();
    void                draw();
    void	        set_coords();
    unsigned int        axis();
    Painter *           painter_;
    NrrdVolume *	volume_;

    bool		nrrd_dirty_;
    bool		tex_dirty_;
    bool		geom_dirty_;

    Point               pos_;
    Vector              xdir_;
    Vector              ydir_;
    //    int                 axis_;
    Plane               plane_;

    ColorMappedNrrdTextureObj *    texture_;
  };
  typedef vector<NrrdSlice *>		NrrdSlices;
  typedef vector<NrrdSlices>		NrrdVolumeSlices;
  typedef map<NrrdVolume *, NrrdSlice*> VolumeSliceMap;

  struct SliceWindow { 
    SliceWindow(Painter *painter, GuiContext *ctx);
    void                setup_gl_view();
    void                push_gl_2d_view();
    void                pop_gl_2d_view();
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
    void		render_frame(double,double,double,double,
                                     double *color1 = 0, double *color2=0);
    void		render_guide_lines(Point);
    void		render_progress_bar();

    Painter *           painter_;
    string		name_;
    WindowLayout *	layout_;
    OpenGLViewport *	viewport_;
    NrrdSlices		slices_;
    VolumeSliceMap      slice_map_;
    NrrdSlice*          paint_layer_;

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
      painter_(painter), name_(name), err_msg_() {}
    virtual ~PainterTool() {};

    enum {
      ERROR_E = 0,
      FALLTHROUGH_E = 1,
      HANDLED_E = 2,
      QUIT_E = 3
    };

    const string &      get_name()      { return name_; }
    const string &      err() { return err_msg_; }

    virtual int         draw(SliceWindow &) { return 0; }
    virtual int         draw_mouse_cursor(Event &) { return 0; }
    
    virtual int         do_event(Event &) { return 0; }

  protected:
    Painter *           painter_;
    string              name_;
    string              err_msg_;
  };

  class CLUTLevelsTool : public PainterTool {
  public:
    CLUTLevelsTool(Painter *painter);
    virtual int         do_event(Event &);
  private:
    Event               press_event_;
    double              scale_;
    double              ww_;
    double              wl_;
  };


  class ZoomTool : public PainterTool {
  public:
    ZoomTool(Painter *painter);
    virtual int         do_event(Event &);
  private:
    Event               press_event_;
    double              zoom_;
  };

  class AutoviewTool : public PainterTool {
  public:
    AutoviewTool(Painter *painter);
    virtual int         do_event(Event &);
  };

  class ProbeTool : public PainterTool {
  public:
    ProbeTool(Painter *painter);
    virtual int         do_event(Event &);
  };


  class PanTool : public PainterTool {
  public:
    PanTool(Painter *painter);
    virtual int         do_event(Event &);
  private:
    Event               press_event_;
    Point               center_;
  };

  class LayerMergeTool : public PainterTool {
  public:
    LayerMergeTool(Painter *painter);
  };

  class CropTool : public PainterTool {
  public:
    CropTool(Painter *painter);
    ~CropTool();
    int                 do_event(Event &);
    int                 draw(SliceWindow &window);
  private:
    typedef vector<BBox> BBoxes;
    void                pick_mouse_motion(Event &);
    void                set_window_cursor(SliceWindow &window, int cursor);
    
    int                 pick_;
    vector<int>         minmax_[2];
    vector<int>         pick_minmax_[2];
    vector<double>	pick_index_;
    double              pick_dist_[2][3];
  };


  class BrushTool : public PainterTool {
  public:
    BrushTool(Painter *painter);
    ~BrushTool();
    int                 do_event(Event &);

    int                 draw_mouse_cursor(Event &);
  private:
    int                 button_press(Event &);
    int                 button_release(Event &);
    int                 my_mouse_motion(Event &);
    int                 key_press(Event &);

    void                line(Nrrd *, double, int, int, int, int, bool);
    void                splat(Nrrd *, double,int,int);
    SliceWindow *       window_;
    NrrdSlice *         slice_;
    float               value_;
    vector<int>         last_index_;
    double              radius_;
  };

  class FloodfillTool : public PainterTool {
  public:
    FloodfillTool(Painter *painter);
    ~FloodfillTool();
    int                 do_event(Event &);
    int                 draw(SliceWindow &window);
  private:
    void                do_floodfill();
    double              value_;
    double              min_;
    double              max_;
    Point               start_pos_;
  };


  class ITKThresholdTool : public PainterTool {
  public:
    ITKThresholdTool(Painter *painter);
    int                 do_event(Event &);
  private:
    NrrdVolume *        seed_volume_;
    NrrdVolume *        source_volume_;
  };


  class ITKConfidenceConnectedImageFilterTool : public PainterTool {
  public:
    ITKConfidenceConnectedImageFilterTool(Painter *painter);
    int                 do_event(Event &);
    int                 draw(SliceWindow &window);
  private:
    vector<int>         seed_;
    NrrdVolume *        volume_;
  };

  class ITKGradientMagnitudeTool : public PainterTool {
  public:
    ITKGradientMagnitudeTool(Painter *painter);
  private:
  };


  class ITKCurvatureAnisotropicDiffusionTool : public PainterTool {
  public:
    ITKCurvatureAnisotropicDiffusionTool(Painter *painter);
  private:
  };

  class ITKBinaryDilateErodeTool : public PainterTool {
  public:
    ITKBinaryDilateErodeTool(Painter *painter);
  private:
  };



  class StatisticsTool : public PainterTool {
  public:
    StatisticsTool(Painter *painter);
    int                 do_event(Event &);
    int                 draw(SliceWindow &window);
  private:
    double              standard_deviation_;
    double              mean_;
    double              sum_;
    double              squared_sum_;
    int                 count_;
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
    // Constructor
    NrrdVolume		(GuiContext *ctx, 
                         const string &name,
                         NrrdDataHandle &);
    // Copy Constructor
    NrrdVolume          (NrrdVolume *copy, 
                         const string &name, 
                         int mode = 0); // if 1, clears out volume
    ~NrrdVolume();
    void                set_nrrd(NrrdDataHandle &);
    NrrdDataHandle      get_nrrd();
    Point               index_to_world(const vector<int> &index);
    Point		index_to_point(const vector<double> &index);
    vector<int>         world_to_index(const Point &p);
    vector<double>      point_to_index(const Point &p);
    vector<double>      vector_to_index(const Vector &v);
    Vector              index_to_vector(const vector<double> &);
    void                build_index_to_world_matrix();
    bool                index_valid(const vector<int> &index);
    template<class T>
    void                get_value(const vector<int> &index, T &value);
    template<class T>
    void                set_value(const vector<int> &index, T value);

    Point               center(int axis = -1, int slice = -1);
    Point               min(int axis = -1, int slice = -1);
    Point               max(int axis = -1, int slice = -1);

    Vector              scale();
    double              scale(unsigned int axis);

    vector<int>         max_index();
    int                 max_index(unsigned int axis);

    bool                inside_p(const Point &p);
    NrrdDataHandle	nrrd_handle_;
    GuiContext *        gui_context_;
    GuiString           name_;
    string              name_prefix_;
    UIdouble		opacity_;
    UIdouble            clut_min_;
    UIdouble            clut_max_;
    Mutex               mutex_;
    float               data_min_;
    float               data_max_;
    GuiInt              colormap_;
    vector<int>         stub_axes_;
    DenseMatrix         transform_;
    bool                keep_;
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
  typedef list<string>                  NrrdVolumeOrder;


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
  NrrdVolumeOrder       volume_order_;
  NrrdVolume *          current_volume_;
  NrrdVolume *          undo_volume_;

  typedef map<string, ColorMapHandle> colormap_map_t;

  colormap_map_t	colormaps_;
  vector<string>        colormap_names_;
  typedef vector<PainterTool *> Tools_t;
  Tools_t               tools_;
  Event                 event_;
  string                filter_text_;

  int			cur_slice_[3];

  UIint			anatomical_coordinates_;
  UIint			show_grid_;
  UIint			show_text_;
  UIdouble		font_r_;
  UIdouble		font_g_;
  UIdouble		font_b_;
  UIdouble		font_a_;

  //! Ports
  BundleOPort *         bundle_oport_;
  typedef vector<BundleHandle> Bundles;


  FreeTypeLibrary *	freetype_lib_;
  typedef		map<string, FreeTypeFace *> FontMap;
  FontMap		fonts_;
  UIdouble		font_size_;
  TextRenderer *        font1_;
  TextRenderer *        font2_;
  TextRenderer *        font3_;
  
  
  RealDrawer *		runner_;
  Thread *		runner_thread_;
  int                   filter_;

  // Methods for drawing to the GL window
  void			redraw_all();
  int			redraw_window(SliceWindow &);

  void			draw_slice_lines(SliceWindow &);

  
  // Methods to render TrueType text labels
  void			initialize_fonts();
  void			delete_all_fonts();
  void			set_font_sizes(double size);



  int			extract_window_slices(SliceWindow &);

  // Methods for navigating around the slices
  void			set_axis(SliceWindow &, unsigned int axis);

  // Methods called by tcl_command
  void                  update_event_state(GuiArgs &args);
  void			handle_gui_mouse_button_press(GuiArgs &args);
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

  NrrdVolume *          create_volume(string name, int mode, int nrrdType);
  void                  recompute_volume_list();  
  void                  show_volume(const string & );
  void                  hide_volume(const string & );
  ColorMapHandle        get_colormap(int);
  void                  send_data();
  void                  receive_filter_bundles(Bundles &);
  void                  receive_normal_bundles(Bundles &);
  void                  extract_data_from_bundles(Bundles &);
  void                  layer_up();
  void                  layer_down();
  void                  create_undo_volume();
  void                  undo_volume();
  pair<double, double>  compute_mean_and_deviation(Nrrd *, Nrrd *);
  NrrdVolume *          copy_current_volume(const string &, int mode=0);
public:
  Painter(GuiContext* ctx);
  virtual ~Painter();
  virtual void		execute();
  virtual void		tcl_command(GuiArgs& args, void*);
  void			real_draw_all();

  static bool           static_callback(void *);

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
  ASSERT((unsigned int)(index.size()) == nrrd->dim);
  int position = index[0];
  int mult_factor = nrrd->axis[0].size;
  for (unsigned int a = 1; a < nrrd->dim; ++a) {
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
  ASSERT((unsigned int)(index.size()) == nrrd->dim);
  int position = index[0];
  int mult_factor = nrrd->axis[0].size;
  for (unsigned int a = 1; a < nrrd->dim; ++a) {
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


int nrrd_type_size(Nrrd *);
int nrrd_data_size(Nrrd *);


template<class T>
void
Painter::NrrdVolume::get_value(const vector<int> &index, T &value) {
  ASSERT(index_valid(index));
  nrrd_get_value(nrrd_handle_->nrrd_, index, value);
}


template <class T>
void
Painter::NrrdVolume::set_value(const vector<int> &index, T value) {
  ASSERT(index_valid(index));
  nrrd_set_value(nrrd_handle_->nrrd_, index, value);
}

template <class T>
unsigned int max_vector_magnitude_index(vector<T> array) {
  if (array.empty()) return 0;
  unsigned int index = 0;
  for (unsigned int i = 1; i < array.size(); ++i) 
    if (fabs(array[i]) > fabs(array[index]))
      index = i;
  return index;
}

}

#endif
