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
#include <Core/Geom/NrrdTextureObj.h>
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


namespace SCIRun {

class Painter : public Module
{
  class SliceWindow;
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
    string *            mouse_motion(MouseState &);
  private:
    double              zoom_;
    SliceWindow *       window_;
  };

  class AutoviewTool : public PainterTool {
  public:
    AutoviewTool(Painter *painter);
    string *            mouse_button_press(MouseState &);
  };

  class ProbeTool : public PainterTool {
  public:
    ProbeTool(Painter *painter);
    string *            mouse_button_press(MouseState &);
    string *            mouse_motion(MouseState &);
  };


  class PanTool : public PainterTool {
  public:
    PanTool(Painter *painter);
    string *            mouse_button_press(MouseState &);
    string *            mouse_motion(MouseState &);
  private:
    double              x_;
    double              y_;
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



  friend class PainterTool;

  enum DisplayMode_e {
    normal_e,
    slab_e,
    mip_e,
    num_display_modes_e
  };


  struct NrrdVolume { 
    NrrdVolume		(GuiContext*ctx);
    NrrdDataHandle	nrrd_;
    GuiString           name_;
    UIdouble		opacity_;
    UIdouble            clut_min_;
    UIdouble            clut_max_;
    Semaphore           semaphore_;
    float               data_min_;
    float               data_max_;
  };

  struct SliceWindow;
  struct WindowLayout;

  struct NrrdSlice {
    NrrdSlice(NrrdVolume *, SliceWindow *);
    string		name_;
    NrrdVolume *	volume_;
    SliceWindow	*	window_;
    NrrdDataHandle      nrrd_;

    int			axis_;
    int			slice_num_;
    int			slab_min_;
    int			slab_max_;
    
    bool		nrrd_dirty_;
    bool		tex_dirty_;
    bool		geom_dirty_;

    unsigned int	mode_;
    unsigned int	tex_wid_;
    unsigned int	tex_hei_;
    unsigned int	wid_;
    unsigned int	hei_;

    float		tex_coords_[8];  // s,t * 4 corners
    float		pos_coords_[12]; // x,y,z * 4 corners
    GLuint		tex_name_;

    NrrdTextureObj *    texture_;


    //    Mutex		lock_;
    //    Thread *	owner_;
    //    int		lock_count_;
    void		do_lock();
    void		do_unlock();
  };
  typedef vector<NrrdSlice *>		NrrdSlices;
  typedef vector<NrrdSlices>		NrrdVolumeSlices;

  struct SliceWindow { 
    SliceWindow(GuiContext *ctx);

    string		name_;
    WindowLayout *	layout_;
    OpenGLViewport *	viewport_;
    NrrdSlices		slices_;

    NrrdSlice		paint_under_;
    NrrdSlice		paint_;
    NrrdSlice		paint_over_;

    UIint		slice_num_;
    UIint		axis_;
    UIdouble		zoom_;
    UIint		slab_min_;
    UIint		slab_max_;
      
    UIdouble		x_;
    UIdouble		y_;

    bool		redraw_;
      
    UIint		auto_levels_;
    UIint		mode_;
    UIint		crosshairs_;
    UIint		snoop_;
    UIint		invert_;
    UIint		reverse_;
      
    UIint		show_guidelines_;
    UIdouble		fusion_;
    
    int			cursor_pixmap_;

    GLdouble		gl_modelview_matrix_[16];
    GLdouble		gl_projection_matrix_[16];
    GLint		gl_viewport_[4];
  };
  typedef vector<SliceWindow *>	SliceWindows;

  struct WindowLayout {
    WindowLayout	(GuiContext *ctx);
    TkOpenGLContext *	opengl_;
    SliceWindows	windows_;
    string		name_;
  };
  typedef map<string, WindowLayout *>	WindowLayouts;

  typedef vector<NrrdVolume *>		NrrdVolumes;




  class RealDrawer : public Runnable {
    Painter *	module_;
    TimeThrottle	throttle_;
  public:
    bool		dead_;
    RealDrawer(Painter* module) : module_(module), throttle_(), dead_(0) {};
    virtual ~RealDrawer();
    virtual void run();
  };
  
  WindowLayouts		layouts_;
  NrrdVolumes		volumes_;
  NrrdDataHandle	gradient_;
  ColorMap2Handle	cm2_;

  NrrdVolume *          current_volume_;


  vector<int>		nrrd_generations_;
  int			cm2_generation_;
  Array3<float>		cm2_buffer_under_;
  Array3<float>		cm2_buffer_;
  Array3<float>		cm2_buffer_over_;

  typedef map<string, ColorMapHandle> colormap_map_t;

  colormap_map_t	colormaps_;
  int			colormap_generation_;


  NrrdSlice *		mip_slices_[3];

  PainterTool *         tool_;
  
  MouseState            mouse_;


  int			pick_;




  int			max_slice_[3];
  int			cur_slice_[3];
  int			slab_width_[3];
  double		scale_[3];
  double		center_[3];
  UIint			show_colormap2_;
  UIint			painting_;

  UIint			texture_filter_;
  UIint			anatomical_coordinates_;
  UIint			show_text_;
  UIdouble		font_r_;
  UIdouble		font_g_;
  UIdouble		font_b_;
  UIdouble		font_a_;

  UIint			dim0_;
  UIint			dim1_;
  UIint			dim2_;
  UIint			geom_flushed_;

  UIdouble		background_threshold_;
  UIdouble		gradient_threshold_;

  PaintCM2Widget *	paint_widget_;
  Mutex			paint_lock_;

  float *		temp_tex_data_;

  //! Ports
  BundleOPort *         bundle_oport_;
  vector<BundleHandle>  bundles_;

  typedef		map<string,TexSquare*> TexSquareMap;
  TexSquareMap		tobjs_;
  map<string,GeomID>    gobjs_;

  FreeTypeLibrary *	freetype_lib_;
  typedef		map<string, FreeTypeFace *> FontMap;
  FontMap		fonts_;
  UIdouble		font_size_;
  
  RealDrawer *		runner_;
  Thread *		runner_thread_;
  CrowdMonitor		slice_lock_;

  // Methods for drawing to the GL window
  void			redraw_all();
  int			redraw_window(SliceWindow &);
  void			setup_gl_view(SliceWindow &);
  int			draw_slice(NrrdSlice &);
  void			bind_slice(NrrdSlice &,float *tex=0,bool filter=true);
  bool			bind_nrrd(Nrrd &);
  void			draw_slice_quad(NrrdSlice &);
  void			draw_guide_lines(SliceWindow &, float, float, float);
  void			draw_slice_lines(SliceWindow &);
  void			draw_slice_arrows(SliceWindow &);
  void			draw_dark_slice_regions(SliceWindow &);


  // Methods to render TrueType text labels
  void			initialize_fonts();
  void			delete_all_fonts();
  void			set_font_sizes(double size);
  void			draw_all_labels(SliceWindow &);
  void			draw_window_label(SliceWindow &);
  void			draw_orientation_labels(SliceWindow &);
  void			draw_position_label(SliceWindow &);
  void			draw_label(SliceWindow &, string, int, int, 
				   FreeTypeText::anchor_e, 
				   FreeTypeFace *font = 0); 

  // Slice extraction and colormapping
  float *		apply_colormap(NrrdSlice &, float *);
  template <class T> 
  void			apply_colormap_to_raw_data(float *, T *, int, int,
						   const float *, int,
						   double, double);
  void			set_slice_coords(NrrdSlice &slice, bool origin);
  int			extract_window_slices(SliceWindow &);
  int			extract_slice(NrrdSlice &);
  int			extract_mip_slices(NrrdVolume *);

  // Methods to send geometry to Viewer
  int			send_mip_textures(SliceWindow &);
  int			send_slice_textures(NrrdSlice &slice);
  void			send_all_geometry();

  // Misc
  void			update_background_threshold();
  double		get_value(const Nrrd *, int, int, int);

  // Methods for navigating around the slices
  void			set_axis(SliceWindow &, unsigned int axis);
  void			next_slice(SliceWindow &);
  void			prev_slice(SliceWindow &);
  void			zoom_in(SliceWindow &);
  void			zoom_out(SliceWindow &);

  // Methods for cursor/coordinate projection and its inverse
  int			x_axis(SliceWindow &);
  int			y_axis(SliceWindow &);
  Point			world_to_screen(SliceWindow &, Point &);
  Point			screen_to_world(SliceWindow &,
					unsigned int x, unsigned int y);

  // Methods called by tcl_command
  void                  update_mouse_state(GuiArgs &args, bool reset = false);
  void			handle_gui_mouse_motion(GuiArgs &args);
  void			handle_gui_mouse_button_press(GuiArgs &args);
  void			handle_gui_mouse_button_release(GuiArgs &args);
  void			handle_gui_keypress(GuiArgs &args);
  void			handle_gui_mouse_enter(GuiArgs &args);
  void			handle_gui_mouse_leave(GuiArgs &args);

  // Methods for Painting into 2D Transfer Function
  void			undo_paint_stroke();
  int			extract_window_paint(SliceWindow &window);
  int			extract_current_paint(SliceWindow &window);
  void			apply_colormap2_to_slice(Array3<float> &, NrrdSlice &);
  bool			rasterize_colormap2();
  void			rasterize_widgets_to_cm2(int min, int max, Array3<float> &);
  void			do_paint(SliceWindow &);

  typedef int (SCIRun::Painter::* SliceWindowFunc)(SliceWindow &);
  typedef int (SCIRun::Painter::* WindowLayoutFunc)(WindowLayout &);
  typedef int (SCIRun::Painter::* NrrdSliceFunc)(NrrdSlice &);
  template <class T> int			for_each(T);
  int			for_each(SliceWindow &, NrrdSliceFunc);
  int			for_each(WindowLayout &, NrrdSliceFunc);
  int			for_each(WindowLayout &, SliceWindowFunc);

  int			render_window(SliceWindow &);
  int			swap_window(SliceWindow &);
  int			set_paint_dirty(SliceWindow &);
  int			autoview(SliceWindow &);

  int			setup_slice_nrrd(NrrdSlice &);
  int			rebind_slice(NrrdSlice &);
  int			set_slice_nrrd_dirty(NrrdSlice &);
  int			update_slice_from_window(NrrdSlice &);

  int                   set_probe(SliceWindow &window);

  int                  create_volume(NrrdVolumes *copies = 0);

public:
  Painter(GuiContext* ctx);
  virtual ~Painter();
  virtual void		execute();
  virtual void		tcl_command(GuiArgs& args, void*);
  void			real_draw_all();
  double		fps_;
  WindowLayout *	current_layout_;
  int			executing_;


};

}

#endif
