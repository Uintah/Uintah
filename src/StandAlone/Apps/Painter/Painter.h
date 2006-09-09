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
#include <stdlib.h>
#include <math.h>
#include <map>
#include <typeinfo>
#include <iostream>
#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>

#include <StandAlone/Apps/Painter/UIvar.h>

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
#include <Core/Geom/OpenGLViewport.h>
#include <Core/Geom/FreeType.h>
#include <Core/Geometry/Plane.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/Timer.h>
#include <Core/Util/Environment.h>
#include <Core/Volume/CM2Widget.h>
#include <Core/Skinner/Parent.h>
#include <Core/Skinner/Color.h>
#include <Core/Skinner/Variables.h>
#include <Core/Events/Tools/BaseTool.h>
#include <Core/Events/Tools/ToolManager.h>
#include <Core/Volume/Texture.h>

#include <include/sci_defs/insight_defs.h>

#ifdef HAVE_INSIGHT
#  include <Core/Datatypes/ITKDatatype.h>
#  include <itkImageToImageFilter.h>
#  include <itkCommand.h>
#  include <itkThresholdSegmentationLevelSetImageFilter.h>
#endif

#ifdef _WIN32
#undef min
#undef max
#endif

namespace SCIRun {

#ifdef HAVE_INSIGHT
using SCIRun::ITKDatatypeHandle;
#endif

class Painter : public Skinner::Parent
{

public:
  class SliceWindow;
private:

  struct Event {
    void                update_state(Painter &painter);
    bool                button(unsigned int) { return 0; }
    bool                shift() { return false; }
    bool                control() { return false; };
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
      MOTION_E    = 1 << 0,
      BUTTON_PRESS_E    = 1 << 1,
      BUTTON_RELEASE_E  = 1 << 2,
      KEY_PRESS_E       = 1 << 3,
      KEY_RELEASE_E     = 1 << 4,
      ENTER_E        = 1 << 5,
      LEAVE_E       = 1 << 6
    };
  };
  
  friend struct Event;
  
  class NrrdVolume;

  struct NrrdSlice { //: public Skinner::Drawable {
    NrrdSlice(Painter *, NrrdVolume *, Point &p, Vector &normal);

    //    propagation_state_e process_event(event_handle_t);
    
    void                bind();
    void                draw();
    void	        set_coords();
    unsigned int        axis();
    void                set_tex_dirty();
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



    
public:
  class SliceWindow : public Skinner::Drawable {
  public:
    SliceWindow(Skinner::Variables *, Painter *painter);
    virtual ~SliceWindow() {}
    
    static string                       class_name() { return "SliceWindow"; }
    propagation_state_e                 process_event(event_handle_t event);

    void                render_gl();
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


    void                set_probe();
    void                extract_slices();
    void                redraw();
    void                autoview(NrrdVolume *, double offset=10.0);
    void                set_axis(unsigned int);

    Painter *           painter_;
    string		name_;
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
private:
  friend struct SliceWindow;

  typedef vector<SliceWindow *>	SliceWindows;

    



  class PainterTool : public BaseTool {
  public:
    PainterTool(Painter *painter, const string &name) : 
      BaseTool(name),
      painter_(painter), name_(name), err_msg_() {}
    virtual ~PainterTool() {};

    enum {
      ERROR_E = 0,
      FALLTHROUGH_E = 1,
      HANDLED_E = 2,
      QUIT_E = 3,
      QUIT_ALL_E = 4
    };

    const string &      get_name()      { return name_; }
    const string &      err() { return err_msg_; }

    virtual int         draw(SliceWindow &) { return 0; }
    virtual int         draw_mouse_cursor(Event &) { return 0; }
    
    virtual int         do_event(Event &) { return 0; }
    virtual propagation_state_e process_event(event_handle_t);

  protected:
    Painter *           painter_;
    string              name_;
    string              err_msg_;
  };

  class PainterPointerTool : public PointerTool {
  public:
    PainterPointerTool(Painter *painter, const string &name) : 
      BaseTool(name),
      PointerTool(name),
      painter_(painter) 
    {
      ASSERT(painter_);
    }
    virtual ~PainterPointerTool() {}

    propagation_state_e pointer_down(int, int, int, unsigned int, int) 
    { 
      return CONTINUE_E; 
    }
    propagation_state_e pointer_up(int, int, int, unsigned int, int) 
    { 
      return QUIT_AND_STOP_E; 
    }
    propagation_state_e pointer_motion(int, int, int, unsigned int, int) {
      return CONTINUE_E;
    }
  protected:
    Painter *           painter_;
  };


  class PointerToolSelectorTool : public PainterPointerTool {
  public:
    PointerToolSelectorTool(Painter *painter);
    virtual ~PointerToolSelectorTool();
    propagation_state_e pointer_down(int, int, int, unsigned int, int);
    propagation_state_e pointer_up(int, int, int, unsigned int, int);
    propagation_state_e pointer_motion(int, int, int, unsigned int, int);
  protected:
    ToolManager &       tm_;
  };

  class KeyToolSelectorTool : public KeyTool {
  public:
    KeyToolSelectorTool(Painter *painter);
    virtual ~KeyToolSelectorTool();
    //    propagation_state_e process_event(event_handle_t);
    propagation_state_e key_press(string, int, unsigned int, unsigned int);
    propagation_state_e key_release(string, int, unsigned int, unsigned int);
  protected:
    Painter *           painter_;
    ToolManager &       tm_;
  };
    

  class CLUTLevelsTool : public PainterPointerTool {
  public:
    CLUTLevelsTool(Painter *painter);
    propagation_state_e pointer_down(int, int, int, unsigned int, int);
    propagation_state_e pointer_motion(int, int, int, unsigned int, int);

  private:
    double              scale_;
    double              ww_;
    double              wl_;
    int                 x_;
    int                 y_;
  };


  class ZoomTool : public PainterPointerTool {
  public:
    ZoomTool(Painter *painter);
    propagation_state_e pointer_down(int, int, int, unsigned int, int);
    propagation_state_e pointer_motion(int, int, int, unsigned int, int);
  private:
    SliceWindow *       window_;
    double              zoom_;
    int                 x_;
    int                 y_;
  };

  class AutoviewTool : public PainterPointerTool {
  public:
    AutoviewTool(Painter *painter);
    propagation_state_e pointer_down(int, int, int, unsigned int, int);
  };

  class ProbeTool : public PainterPointerTool {
  public:
    ProbeTool(Painter *painter);
    propagation_state_e pointer_down(int, int, int, unsigned int, int);
    propagation_state_e pointer_motion(int, int, int, unsigned int, int);
  };


  class PanTool : public PainterPointerTool {
  public:
    PanTool(Painter *painter);
    propagation_state_e pointer_down(int, int, int, unsigned int, int);
    propagation_state_e pointer_motion(int, int, int, unsigned int, int);
  private:
    int                 x_;
    int                 y_;
    Point               center_;
    SliceWindow *       window_;
  };

  class LayerMergeTool : public PainterTool {
  public:
    LayerMergeTool(Painter *painter);
  };

  class CropTool : public virtual BaseTool,
                   public PointerTool {
  public:
    CropTool(Painter *painter);
    ~CropTool();
    propagation_state_e process_event(event_handle_t);
    propagation_state_e pointer_motion(int b, int x, int y,
                                     unsigned m, int t);
    propagation_state_e pointer_down(int b, int x, int y,
                                     unsigned m, int t);
    propagation_state_e pointer_up(int b, int x, int y,
                                   unsigned m, int t);
  private:
    int                 draw_gl(SliceWindow &window);
    typedef vector<BBox> BBoxes;
    void                finish();
    //    void                pick_mouse_motion(Event &);
    void                set_window_cursor(SliceWindow &window, int cursor);
    void                update_to_gui();
    void                update_from_gui();
    Painter *           painter_;
    int                 pick_;
    vector<int>         minmax_[2];
    vector<int>         pick_minmax_[2];
    vector<double>	pick_index_;
    double              pick_dist_[2][3];
  };


  class BrushTool : virtual public BaseTool,
                    public PointerTool {
  public:
    BrushTool(Painter *painter);
    ~BrushTool();
    propagation_state_e process_event(event_handle_t);
    propagation_state_e pointer_motion(int b, int x, int y,
                                     unsigned m, int t);

    propagation_state_e pointer_down(int b, int x, int y,
                                     unsigned m, int t);
    
    propagation_state_e pointer_up(int b, int x, int y,
                                   unsigned m, int t);

    propagation_state_e key_press(string key, int keyval,
                                  unsigned int modifiers, unsigned int time);

      //    int                 draw_mouse_cursor(Event &);
  private:
    void                draw_gl(SliceWindow &);
    void                line(Nrrd *, double, int, int, int, int, bool);
    void                splat(Nrrd *, double,int,int);

    Painter *           painter_;
    SliceWindow *       window_;
    NrrdSlice *         slice_;
    float               value_;
    vector<int>         last_index_;
    double              radius_;
    bool                draw_cursor_;
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

#ifdef HAVE_INSIGHT
  typedef             itk::Image<float,3> ITKImageFloat3D;
  class ITKThresholdTool : public BaseTool {
  public:
    ITKThresholdTool(Painter *painter);
    propagation_state_e process_event(event_handle_t);
  private:
    void                finish();
    void                cont();
    void                set_vars();
    Painter *           painter_;
    NrrdVolume *        seed_volume_;
    

    typedef itk::ThresholdSegmentationLevelSetImageFilter
    < Painter::ITKImageFloat3D, Painter::ITKImageFloat3D > FilterType;
    FilterType::Pointer filter_;
    
  };
#endif

  class ITKConfidenceConnectedImageFilterTool : public virtual BaseTool,
                                                public PainterPointerTool
  {
  public:
    ITKConfidenceConnectedImageFilterTool(Painter *painter);
    propagation_state_e pointer_down(int, int, int, unsigned int, int);
    propagation_state_e pointer_up(int, int, int, unsigned int, int);
    propagation_state_e pointer_motion(int, int, int, unsigned int, int);
    propagation_state_e process_event(event_handle_t);
  private:
    void                draw_gl(SliceWindow &window);
    void                finish();
    vector<int>         seed_;
    NrrdVolume *        volume_;
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
  friend class BaseTool;

  enum DisplayMode_e {
    normal_e,
    slab_e,
    mip_e,
    num_display_modes_e
  };

  class NrrdVolume { 
  public:
    // Constructor
    NrrdVolume		(VarContext *ctx, 
                         const string &name,
                         NrrdDataHandle &);
    // Copy Constructor
    NrrdVolume          (NrrdVolume *copy, 
                         const string &name, 
                         int mode = 0); // if 1, clears out volume
    ~NrrdVolume();
    void                set_nrrd(NrrdDataHandle &);
    void                reset_data_range();
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
    VarContext *        gui_context_;
    //    GuiString           name_;
    string              name_;
    string              name_prefix_;
    UIdouble		opacity_;
    UIdouble            clut_min_;
    UIdouble            clut_max_;
    Mutex               mutex_;
    float               data_min_;
    float               data_max_;
    int                 colormap_;
    vector<int>         stub_axes_;
    DenseMatrix         transform_;
    bool                keep_;
  };


  typedef vector<NrrdVolume *>		NrrdVolumes;
  typedef map<string, NrrdVolume *>	NrrdVolumeMap;
  typedef list<string>                  NrrdVolumeOrder;



  SliceWindow *         cur_window_;
  ToolManager           tm_;
  Point                 pointer_pos_;

  SliceWindows		windows_;
  NrrdVolumes		volumes_;
  NrrdVolumeMap         volume_map_;
  NrrdVolumeOrder       volume_order_;
  NrrdVolume *          current_volume_;
  NrrdVolume *          undo_volume_;

  typedef map<string, ColorMapHandle> colormap_map_t;

  colormap_map_t	colormaps_;
  vector<string>        colormap_names_;
  typedef vector<PainterTool *> Tools_t;

  // todo delete
  Tools_t               tools_;
  // todo delete
  Event                 event_;
  // todo delete
  string                filter_text_;

  UIint			anatomical_coordinates_;
  UIint			show_grid_;
  UIint			show_text_;

  Mutex                 volume_lock_;

  typedef vector<BundleHandle> Bundles;
  Bundles               bundles_;

  TextureHandle         volume_texture_;
  // Methods for drawing to the GL window
  void			redraw_all();

  void			draw_slice_lines(SliceWindow &);

  
  
  void			extract_all_window_slices();

  void                  set_probe();
  void                  copy_current_layer();
  void                  kill_current_layer();
  void                  new_current_layer();
  void                  set_all_slices_tex_dirty();

  NrrdVolume *          create_volume(string name, int mode, int nrrdType);
  void                  recompute_volume_list();  
  void                  show_volume(const string & );
  void                  hide_volume(const string & );
  ColorMapHandle        get_colormap(int);
  void                  send_data();

  void                  receive_filter_bundles(Bundles &);
  void                  receive_normal_bundles(Bundles &);
  void                  extract_data_from_bundles(Bundles &);

  void                  move_layer_up();
  void                  move_layer_down();
  void                  cur_layer_up();
  void                  cur_layer_down();

  void                  opacity_up();
  void                  opacity_down();

  void                  reset_clut();

  void                  create_undo_volume();
  void                  undo_volume();
  pair<double, double>  compute_mean_and_deviation(Nrrd *, Nrrd *);
  NrrdVolume *          copy_current_volume(const string &, int mode=0);

  NrrdVolume *          filter_volume_;
  bool                  abort_filter_;
#ifdef HAVE_INSIGHT
  ITKDatatypeHandle     filter_update_img_;
  ITKDatatypeHandle     nrrd_to_itk_image(NrrdDataHandle &nrrd);
  //itk::Object::Pointer  nrrd_to_itk_image(NrrdDataHandle &nrrd);
  
  NrrdDataHandle        itk_image_to_nrrd(ITKDatatypeHandle &);


  template <class ImageT>
  bool                  do_itk_filter(itk::ImageToImageFilter<ImageT,ImageT> *,
                                      NrrdDataHandle &nrrd);

  void                  filter_callback(itk::Object *, 
                                        const itk::EventObject &);
  void                  filter_callback_const (const itk::Object *, 
                                               const itk::EventObject &);
#endif
  
  //  propagation_state_e   start_brush_tool(event_handle_t);
  CatcherFunction_t     InitializeSignalCatcherTargets;
  CatcherFunction_t     SliceWindow_Maker;

  CatcherFunction_t     StartBrushTool;
  CatcherFunction_t     StartCropTool;
  CatcherFunction_t     StartFloodFillTool;

  CatcherFunction_t     Autoview;
  CatcherFunction_t     CopyLayer;
  CatcherFunction_t     DeleteLayer;
  CatcherFunction_t     NewLayer;
  CatcherFunction_t     MergeLayer;

  CatcherFunction_t     MemMapFileRead;
  CatcherFunction_t     NrrdFileRead;
  CatcherFunction_t     NrrdFileWrite;

  CatcherFunction_t     FinishTool;
  CatcherFunction_t     CancelTool;
  CatcherFunction_t     SetLayer;
  CatcherFunction_t     LoadColorMap1D;

  CatcherFunction_t     ITKBinaryDilate;
  CatcherFunction_t     ITKImageFileRead;
  CatcherFunction_t     ITKImageFileWrite;
  CatcherFunction_t     ITKGradientMagnitude;
  CatcherFunction_t     ITKBinaryDilateErode;
  CatcherFunction_t     ITKCurvatureAnisotropic;
  CatcherFunction_t     ITKConfidenceConnected;
  CatcherFunction_t     ITKThresholdLevelSet;

  CatcherFunction_t     ShowVolumeRendering;
  CatcherFunction_t     AbortFilterOn;
  CatcherFunction_t     ResampleVolume;

  


public:
  static Skinner::DrawableMakerFunc_t maker;
  static string         class_name() { return "Painter"; }
  virtual int           get_signal_id(const string &signalname);


  Painter(Skinner::Variables *, VarContext* ctx);
  virtual ~Painter();
  //  virtual void		tcl_command(GuiArgs& args, void*);
  void			add_bundle(BundleHandle);  
};




class RedrawSliceWindowEvent : public RedrawEvent 
{
  Painter::SliceWindow &        window_;
public:
  RedrawSliceWindowEvent(Painter::SliceWindow &window) :
    RedrawEvent(),
    window_(window)
  {
  }
  
  ~RedrawSliceWindowEvent() {}
  Painter::SliceWindow &       get_window() { return window_; }
};



class FinishEvent : public QuitEvent 
{
public:
  FinishEvent() : QuitEvent() {}
  ~FinishEvent() {}
};

class SetLayerEvent : public RedrawEvent 
{
public:
  SetLayerEvent() : RedrawEvent() {}
  ~SetLayerEvent() {}
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


#ifdef HAVE_INSIGHT

template <class ImageType>
bool
Painter::do_itk_filter(itk::ImageToImageFilter<ImageType, ImageType> *filter,
                       NrrdDataHandle &nrrd_handle) 
{
  typedef typename itk::MemberCommand< Painter > RedrawCommandType;
  typename RedrawCommandType::Pointer callback = RedrawCommandType::New();
  callback->SetCallbackFunction(this, &Painter::filter_callback);
  callback->SetCallbackFunction(this, &Painter::filter_callback_const);
  filter->AddObserver(itk::ProgressEvent(), callback);
  filter->AddObserver(itk::IterationEvent(), callback);
  
  if (nrrd_handle.get_rep()) {
    ITKDatatypeHandle img_handle = nrrd_to_itk_image(nrrd_handle);  
    ImageType *imgp = dynamic_cast<ImageType *>(img_handle->data_.GetPointer());
    
    if (imgp == 0) 
      return false;
    
    filter->SetInput(imgp);
  }

  try {
    filter->Update();
  } catch (itk::ExceptionObject &err) {
    if (!abort_filter_) {
      cerr << "ITK Exception: \n";
      err.Print(cerr);
      return false;
    } else {
      abort_filter_ = false;
    }
  } catch (...) {
    cerr << "ITK Filter error!\n";
    return false;
  }
  
  SCIRun::ITKDatatypeHandle output_img = new SCIRun::ITKDatatype();
  output_img->data_ = filter->GetOutput();

  nrrd_handle = itk_image_to_nrrd(output_img);

#if 0
  get_vars()->insert("ProgressBar::bar_height","0","string", true);
  get_vars()->insert("Painter::progress_bar_total_width","0","string", true);
  get_vars()->insert("Painter::progress_bar_text","F","string", true);
  get_vars()->insert("Painter::progress_bar_done_width","0","string", true);
  get_vars()->insert("ToolDialog::button_height","0","string", true);
  get_vars()->insert("ToolDialog::text","","string", true);
#endif

  return true;
}
#endif

}
#endif
