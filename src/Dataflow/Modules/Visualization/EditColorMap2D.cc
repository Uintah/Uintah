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
//    EditColorMap2D
//    File   : EditColorMap2D.cc
//    Author : Milan Ikits
//    Author : Michael Callahan
//    Date   : Thu Jul  8 01:50:58 2004

#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Array3.h>
#include <Core/Persistent/Pstreams.h>

#include <sci_gl.h>
#include <Core/Volume/CM2Shader.h>
#include <Core/Volume/CM2Widget.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Dataflow/Ports/Colormap2Port.h>
#include <Dataflow/Ports/NrrdPort.h>

#include <Core/Geom/GeomOpenGL.h>
#include <Core/Geom/TkOpenGLContext.h>
#include <Core/Util/Endian.h>
#include <stdio.h>
#include <stack>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>

// tcl interpreter corresponding to this module
extern Tcl_Interp* the_interp;

using std::stack;

namespace SCIRun {

struct UndoItem
{
  enum Action { UNDO_CHANGE, UNDO_ADD, UNDO_DELETE };
  int				action_;
  int				selected_;
  CM2WidgetHandle		widget_;
  UndoItem(int a, int s, CM2WidgetHandle w)
    : action_(a), selected_(s), widget_(w) {}
};


class EditColorMap2D : public Module {

public:
  EditColorMap2D(GuiContext* ctx);
  virtual ~EditColorMap2D();
  
  // Virtually Inhereted from Dataflow/Modules/Module.h
  virtual void			execute();
  virtual void			tcl_command(GuiArgs&, void*);
  virtual void			presave();
  
private:
  void				add_triangle_widget();
  void				add_rectangle_widget();
  void				add_paint_widget();
  void				delete_selected_widget();
  void				resize_gui(int n = -1);
  void				update_from_gui();
  void				update_to_gui(bool forward = true);
  void				tcl_unpickle();

  void				undo();

  void				save_ppm(const string &filename,
					 int sx, int sy,
					 int bpp, const unsigned char * buf);
  void				save_function(const string& filename);

  void				init_shader_factory();
  void				build_colormap_texture();
  void				build_histogram_texture();
  void				draw_texture(GLuint &id);
  void				redraw();

  void				push(int x, int y, int button);
  void				motion(int x, int y);
  void				release(int x, int y);
  void				mouse_pick(int x, int y, int b);
  void				set_window_cursor(int x, int y);
  void				select_widget(int, int);
  void				screen_val(int &x, int &y);
  //! functions for panning.
  void				translate_start(int x, int y);
  void				translate_motion(int x, int y);
  void				translate_end(int x, int y);
  //! functions for zooming.
  void				scale_start(int x, int y);
  void				scale_motion(int x, int y);
  void				scale_end(int x, int y);
  
  // Input/Output Ports
  ColorMap2IPort*		cmap_iport_;
  ColorMap2OPort*		cmap_oport_;
  NrrdIPort*			hist_iport_;
  ColorMap2Handle		sent_cmap2_;

  bool				just_resend_selection_;
  TkOpenGLContext *		ctx_;
  int				width_;
  int				height_;
  int				button_;
  vector<CM2WidgetHandle>	widgets_;
  stack<UndoItem>		undo_stack_;
  CM2ShaderFactory*		shader_factory_;
  Array3<float>			colormap_texture_;
  bool				use_back_buffer_;
  
  int				icmap_generation_;
  int				hist_generation_;

  Nrrd*				histo_;
  bool				histo_dirty_;
  GLuint			histogram_texture_id_;

  bool				cmap_dirty_;
  GLuint			colormap_texture_id_;

  int				mouse_widget_;
  int				mouse_object_;
  // Push on undo when motion occurs, not on select.
  bool				first_motion_;
  bool				save_ppm_;
  
  int				mouse_last_x_;
  int				mouse_last_y_;
  GuiDouble			pan_x_;
  GuiDouble			pan_y_;
  GuiDouble			scale_;

  bool				updating_; // updating the tf or not

  GuiInt			gui_faux_;
  GuiDouble			gui_histo_;

  // The currently selected widget
  GuiInt			gui_selected_widget_;
  // The currently selected widgets selected object
  GuiInt			gui_selected_object_;

  GuiInt			gui_num_entries_;
  vector<GuiString *>		gui_name_;
  vector<GuiDouble *>		gui_color_r_;
  vector<GuiDouble *>		gui_color_g_;
  vector<GuiDouble *>		gui_color_b_;
  vector<GuiDouble *>		gui_color_a_;
  vector<GuiString *>		gui_wstate_;
  vector<GuiInt *>		gui_sstate_;
  vector<GuiInt *>		gui_onstate_;

  // variables for file loading and saving
  GuiFilename			filename_;
  string			old_filename_;
  time_t			old_filemodification_;
  GuiString *			end_marker_;
  double			value_offset_;
  double			value_scale_;
};



DECLARE_MAKER(EditColorMap2D)

EditColorMap2D::EditColorMap2D(GuiContext* ctx)
  : Module("EditColorMap2D", ctx, Filter, "Visualization", "SCIRun"),
    cmap_iport_((ColorMap2IPort*)get_iport("Input Colormap")),
    cmap_oport_((ColorMap2OPort*)get_oport("Output Colormap")),
    hist_iport_((NrrdIPort*)get_iport("Histogram")),
    sent_cmap2_(0),
    just_resend_selection_(0),
    ctx_(0), 
    width_(0),
    height_(0),
    button_(0),
    widgets_(),
    undo_stack_(),
    shader_factory_(0),
    colormap_texture_(256, 512, 4),
    use_back_buffer_(true),
    icmap_generation_(-1),
    hist_generation_(-1),
    histo_(0), 
    histo_dirty_(true), 
    histogram_texture_id_(0),
    cmap_dirty_(true), 
    colormap_texture_id_(0),
    mouse_widget_(-1),
    mouse_object_(0),
    first_motion_(true),
    save_ppm_(false),
    mouse_last_x_(0),
    mouse_last_y_(0),
    pan_x_(ctx->subVar("panx")),
    pan_y_(ctx->subVar("pany")),
    scale_(ctx->subVar("scale_factor")),
    updating_(false),
    gui_faux_(ctx->subVar("faux")),
    gui_histo_(ctx->subVar("histo")),
    gui_selected_widget_(ctx->subVar("selected_widget"), -1),
    gui_selected_object_(ctx->subVar("selected_object"), -1),
    gui_num_entries_(ctx->subVar("num-entries")),
    gui_name_(),
    gui_color_r_(),
    gui_color_g_(),
    gui_color_b_(),
    gui_color_a_(),
    gui_wstate_(),
    gui_sstate_(),
    gui_onstate_(),
    filename_(ctx->subVar("filename")),
    old_filename_(),
    old_filemodification_(0),
    end_marker_(0),
    value_offset_(0.0),
    value_scale_(1.0)
{
  pan_x_.set(0.0);
  pan_y_.set(0.0);
  scale_.set(1.0);
  widgets_.push_back(scinew TriangleCM2Widget());
  widgets_.push_back(scinew RectangleCM2Widget());
  resize_gui(2);
  update_to_gui(false);
}


EditColorMap2D::~EditColorMap2D()
{
  delete shader_factory_;
  if (end_marker_) 
    delete end_marker_;
}


void
EditColorMap2D::translate_start(int x, int y)
{
  mouse_last_x_ = x;
  mouse_last_y_ = y;
}

void
EditColorMap2D::translate_motion(int x, int y)
{
  float xmtn = float(mouse_last_x_ - x) / float(width_);
  float ymtn = -float(mouse_last_y_ - y) / float(height_);
  mouse_last_x_ = x;
  mouse_last_y_ = y;

  pan_x_.set(pan_x_.get() + xmtn / scale_.get());
  pan_y_.set(pan_y_.get() + ymtn / scale_.get());

  redraw();
}

void
EditColorMap2D::translate_end(int x, int y)
{
  redraw();
}

void
EditColorMap2D::scale_start(int x, int y)
{
  mouse_last_y_ = y;
}

void
EditColorMap2D::scale_motion(int x, int y)
{
  float ymtn = -float(mouse_last_y_ - y) / float(height_);
  mouse_last_y_ = y;
  scale_.set(scale_.get() + -ymtn);

  if (scale_.get() < 0.0) scale_.set(0.0);

  redraw();
}

void
EditColorMap2D::scale_end(int x, int y)
{
  redraw();
}

int
do_round(float d)
{
  if (d > 0.0) return (int)(d + 0.5);
  else return -(int)(-d + 0.5); 
}

void
EditColorMap2D::screen_val(int &x, int &y)
{
  const float cx = width_ * 0.5;
  const float cy = height_ * 0.5;
  const float sf_inv = 1.0 / scale_.get();

  x = do_round((x - cx) * sf_inv + cx + (pan_x_.get() * width_));
  y = do_round((y - cy) * sf_inv + cy - (pan_y_.get() * height_));
}

void
EditColorMap2D::tcl_command(GuiArgs& args, void* userdata)
{
  if (args.count() < 2) {
    args.error("No command for EditTransferFunc");
    return;
  }

  if (args[1] == "addtriangle") add_triangle_widget();
  else if (args[1] == "addrectangle") add_rectangle_widget();
  else if (args[1] == "addpaint") add_paint_widget();
  else if (args[1] == "deletewidget") delete_selected_widget();
  else if (args[1] == "undowidget") undo();
  else if (args[1] == "unpickle") tcl_unpickle();
  else if (args[1] == "resend_selection") {
    just_resend_selection_ = true;
    want_to_execute();
  } else if (args[1] == "mouse") {
    int X, Y; // X, Y are unscaled/untranslated coordinates
    string_to_int(args[3], X);
    string_to_int(args[4], Y);

    int x = X, y = Y; // x, y are scaled/translated coordinates
    screen_val(x,y);

    int b;
    string_to_int(args[5], b); // which button it was

    if (args[2] == "motion") motion(x, y);
    else if (args[2] == "push") push(x, y, b);
    else if (args[2] == "release") release(x, y);
    else if (args[2] == "x_late_start") translate_start(X, Y);
    else if (args[2] == "x_late_motion") translate_motion(X, Y);
    else if (args[2] == "x_late_end") translate_end(X, Y);
    else if (args[2] == "scale_start") scale_start(X, Y);
    else if (args[2] == "scale_motion") scale_motion(X, Y);
    else if (args[2] == "scale_end") scale_end(X, Y);
    else if (args[2] == "reset") {
      pan_x_.set(0.0);
      pan_y_.set(0.0);
      scale_.set(1.0);
      redraw();
    }
  } else if (args[1] == "redraw") {
    if (args.count() > 2)
    {
      gui_selected_widget_.reset();
      gui_selected_object_.reset();
      select_widget(gui_selected_widget_.get(), 
		    gui_selected_object_.get());
      cmap_dirty_ = true;
    }
    redraw();
  } else if (args[1] == "destroygl") {
    if (ctx_) {
      delete ctx_;
      ctx_ = 0;
    }
  } else if (args[1] == "setgl") {
    ASSERT(args.count() == 3);
    if (ctx_) {
      delete ctx_;
    }
    ctx_ = scinew TkOpenGLContext(args[2], 0, 512, 256);
    width_ = ctx_->width();
    height_ = ctx_->height();
    width_ = 512;
    height_ = 256;
  } else if (!args[1].compare(0, 13, "color_change-")) {
    int n;
    string_to_int(args[1].substr(13), n);
    gui_num_entries_.reset();
    resize_gui();
    if (n < 0 || n >= gui_num_entries_.get()) return;

    gui_color_r_[n]->reset();
    gui_color_g_[n]->reset();
    gui_color_b_[n]->reset();
    gui_color_a_[n]->reset();
    const double a = gui_color_a_[n]->get();
    Color new_color
      (gui_color_r_[n]->get(), gui_color_g_[n]->get(),  gui_color_b_[n]->get());
    if (new_color != widgets_[n]->color() || a != widgets_[n]->alpha())
    {
      undo_stack_.push(UndoItem(UndoItem::UNDO_CHANGE, n,
				widgets_[n]->clone()));
      widgets_[n]->set_color(new_color);
      widgets_[n]->set_alpha(a);
      want_to_execute();
    }

  } else if (!args[1].compare(0, 12, "shadewidget-")) {
    int n;
    string_to_int(args[1].substr(12), n);
    resize_gui();  // make sure the guivar vector exists
    gui_num_entries_.reset();
    if (n < 0 || n >= gui_num_entries_.get()) return;

    // Toggle the shading type from flat to normal and vice-versa
    gui_sstate_[n]->reset();
    widgets_[n]->set_shadeType(gui_sstate_[n]->get());
    cmap_dirty_ = true;
    redraw();
    want_to_execute();
  } else if(!args[1].compare(0, 9, "toggleon-")) {
    int n;
    string_to_int(args[1].substr(9), n);
    resize_gui();  // make sure the guivar vector exists
    gui_num_entries_.reset();
    if (n < 0 || n >= gui_num_entries_.get()) return;

    gui_onstate_[n]->reset();
    widgets_[n]->set_onState(gui_onstate_[n]->get());  // toggle on/off state.
    
    cmap_dirty_ = true;
    redraw();
    want_to_execute();
  } else if (args[1] == "swatch_save") {
    save_ppm_ = true;
    filename_.reset();
    cmap_dirty_ = true;
    redraw();
  } else if (args[1] == "load") {
    // The implementation of this was taken almost directly from
    // NrrdReader Module.  
    filename_.reset();
    string fn(filename_.get());
    if(fn == "") {
      error("Please Specify a Transfer Function filename.");
      return;
    }

    struct stat buf;
    if(stat(fn.c_str(), &buf) == -1) {
      error(string("EditColorMap2D error - file not found: '")+fn+"'");
      return;
    }

    // if we haven't read yet, or if it's new, or if time has changed, read.
#ifdef __sgi
    time_t new_filemodification = buf.st_mtim.tv_sec;
#else 
    time_t new_filemodification = buf.st_mtime;
#endif
    if( new_filemodification != old_filemodification_)
    {
      old_filemodification_ = new_filemodification;
      old_filename_ = fn;
      
      int len = fn.size();
      const string suffix(".xff");

      // Check the suffix.
      if (fn.substr(len - 4, 4) == suffix) {
        BinaryPiostream *stream = new BinaryPiostream(fn, Piostream::Read);
        if (!stream) {
          error("Error reading file '" + fn + "'.");
          return;
        }

        // read the file.
        int size;
        stream->io(size);
        if(stream->reading()) widgets_.resize(size);
        for(unsigned int i = 0; i < widgets_.size(); i++) {
          Pio(*stream, widgets_[i]);
        }
        delete stream;

        cmap_dirty_ = true;
        redraw();
        update_to_gui();
        want_to_execute();
      }
    }    
  } else if (args[1] == "save") {
    const string fn(filename_.get());
    save_function(fn);

  } else if (args[1] == "saveppm") { 
    save_ppm_ = true;
    cmap_dirty_ = true;
    redraw();
    // Now we trim off the .ppm, add a .xff and call
    // the save_function() method
    string fn = filename_.get();
    fn += ".xff";
    char buf[256];
    sprintf(buf, fn.c_str());
    
    const string xff_file = buf;
    save_function(xff_file);
  }
  else {
    Module::tcl_command(args, userdata);
  }
}


void
EditColorMap2D::add_triangle_widget()
{
    widgets_.push_back(scinew TriangleCM2Widget());
    undo_stack_.push(UndoItem(UndoItem::UNDO_ADD, widgets_.size()-1, NULL));
    update_to_gui();
    select_widget(widgets_.size()-1, 1);
    cmap_dirty_ = true;
    redraw();
    want_to_execute();
}


void
EditColorMap2D::add_rectangle_widget()
{
    widgets_.push_back(scinew RectangleCM2Widget());
    undo_stack_.push(UndoItem(UndoItem::UNDO_ADD, widgets_.size()-1, NULL));
    update_to_gui();
    select_widget(widgets_.size()-1, 1);
    cmap_dirty_ = true;
    redraw();
    want_to_execute();
}

void
EditColorMap2D::add_paint_widget()
{
  widgets_.push_back(scinew PaintCM2Widget());
  undo_stack_.push(UndoItem(UndoItem::UNDO_ADD, widgets_.size()-1, NULL));
  update_to_gui();
  select_widget(widgets_.size()-1, 1);
  // PaintCM2Widgets start blank, so no need set cmap_dirty and redraw here
  want_to_execute();
}

void
EditColorMap2D::delete_selected_widget()
{
  gui_selected_widget_.reset();
  const int widget = gui_selected_widget_.get();
  
  if (widget < 0 || widget >= (int)widgets_.size()) return;
  // Delete widget.
  undo_stack_.push(UndoItem(UndoItem::UNDO_DELETE, widget, widgets_[widget]));
  widgets_.erase(widgets_.begin() + widget);

  update_to_gui();
  if (gui_selected_widget_.get() >= (int)widgets_.size())
    select_widget(widgets_.size()-1, 1);
  else 
    select_widget(gui_selected_widget_.get(), 1);
  cmap_dirty_ = true;
  redraw();
  want_to_execute();
}


void
EditColorMap2D::save_function(const string& filename)
{
  if (filename == "") {
    error("Warning;  No filename provided to EditColorMap2D");
    return;
  }
  
  // Open ostream
  Piostream* stream;
  stream = scinew BinaryPiostream(filename, Piostream::Write);
  if (stream->error())
    error("Could not open file for writing" + filename);
  else {
    int size = widgets_.size();
    Pio(*stream, size);
    for(unsigned int i = 0; i < widgets_.size(); i++) {
      Pio(*stream, widgets_[i]);
    }
    delete stream;
  }
}



void
EditColorMap2D::presave()
{
  unsigned int i;

  resize_gui();
  update_to_gui(false);

  // Pickle up the tcl states.
  for (i = 0; i < widgets_.size(); i++)
  {
    gui_wstate_[i]->set(widgets_[i]->tcl_pickle());
  }

  const unsigned int ws = widgets_.size();
  if (ws < gui_name_.size())
  {
    // Delete all of the unused variables.
    for (i = ws; i < gui_name_.size(); i++)
    {
      delete gui_name_[i];
      delete gui_color_r_[i];
      delete gui_color_g_[i];
      delete gui_color_b_[i];
      delete gui_color_a_[i];
      delete gui_wstate_[i];
      delete gui_sstate_[i];
      delete gui_onstate_[i];
    }

    gui_name_.erase(gui_name_.begin() + ws, gui_name_.end());
    gui_color_r_.erase(gui_color_r_.begin() + ws, gui_color_r_.end());
    gui_color_g_.erase(gui_color_g_.begin() + ws, gui_color_g_.end());
    gui_color_b_.erase(gui_color_b_.begin() + ws, gui_color_b_.end());
    gui_color_a_.erase(gui_color_a_.begin() + ws, gui_color_a_.end());
    gui_wstate_.erase(gui_wstate_.begin() + ws, gui_wstate_.end());
    gui_sstate_.erase(gui_sstate_.begin() + ws, gui_sstate_.end());
    gui_onstate_.erase(gui_onstate_.begin() + ws, gui_onstate_.end());
  }
}


void
EditColorMap2D::undo()
{
  bool gui_update = false;
  if (!undo_stack_.empty())
  {
    const UndoItem &item = undo_stack_.top();
    
    switch (item.action_)
    {
    case UndoItem::UNDO_CHANGE:
      widgets_[item.selected_] = item.widget_;
      gui_update = true;
      break;

    case UndoItem::UNDO_ADD:
      widgets_.erase(widgets_.begin() + item.selected_);
      gui_update = true;
      break;
   
    case UndoItem::UNDO_DELETE:
      widgets_.insert(widgets_.begin() + item.selected_, item.widget_);
      gui_update = true;
      break;
    }
    undo_stack_.pop();
    cmap_dirty_ = true;
    want_to_execute();
  }
}


void
EditColorMap2D::resize_gui(int n)
{
  gui_num_entries_.reset();
  if (gui_num_entries_.get() == (int)widgets_.size()) return;  
  gui_num_entries_.set(n==-1?widgets_.size():n);
  unsigned int i = 0;
  // Expand the gui elements.
  for (i = gui_name_.size(); i < (unsigned int)gui_num_entries_.get(); i++)
  {
    const string num = to_string(i);
    gui_name_.push_back(new GuiString(ctx->subVar("name-" + num)));
    gui_color_r_.push_back(new GuiDouble(ctx->subVar(num +"-color-r")));
    gui_color_g_.push_back(new GuiDouble(ctx->subVar(num +"-color-g")));
    gui_color_b_.push_back(new GuiDouble(ctx->subVar(num +"-color-b")));
    gui_color_a_.push_back(new GuiDouble(ctx->subVar(num +"-color-a")));
    gui_wstate_.push_back(new GuiString(ctx->subVar("state-" + num)));
    gui_sstate_.push_back(new GuiInt(ctx->subVar("shadeType-" + num)));
    gui_onstate_.push_back(new GuiInt(ctx->subVar("on-" + num)));

  }
  // This marker stuff is for TCL, its the last variable created, so
  // its also the last variable written out to the .net script
  // look @ the TCL array ModuleSavedVars(EditColorMap2D_0)
  // First: Delete the old variable that marked the end of the variables
  if (end_marker_) 
    delete end_marker_;
  // Second: Create a new marker that marks the end of the variables
  if (i != 0) 
    end_marker_ = scinew GuiString(ctx->subVar("marker"), "end");
}


void
EditColorMap2D::update_to_gui(bool forward)
{
  // Update GUI
  resize_gui();
  for (unsigned int i = 0; i < widgets_.size(); i++)
  {
    gui_name_[i]->set(widgets_[i]->name());
    Color c(widgets_[i]->color());
    gui_color_r_[i]->set(c.r());
    gui_color_g_[i]->set(c.g());
    gui_color_b_[i]->set(c.b());
    gui_color_a_[i]->set(widgets_[i]->alpha());
    gui_sstate_[i]->set(widgets_[i]->get_shadeType());
    gui_onstate_[i]->set(widgets_[i]->get_onState());
  }
  if (forward) { 
    gui->execute(id + " create_entries"); 
  }
}


void
EditColorMap2D::update_from_gui()
{
  ctx->reset();   // Reset GUI vars cache
  resize_gui();   // Make sure we have enough GUI vars to read through
  for (unsigned int i = 0; i < widgets_.size(); i++)
  {
    if (widgets_[i]->name() != gui_name_[i]->get()) {
      widgets_[i]->name() = gui_name_[i]->get();
      cmap_dirty_ = true;
    }
    Color new_color(gui_color_r_[i]->get(),
		    gui_color_g_[i]->get(),
		    gui_color_b_[i]->get());
    if (widgets_[i]->color() != new_color) {
      widgets_[i]->set_color(new_color);
      cmap_dirty_ = true;
    }
    
    if (widgets_[i]->alpha() != gui_color_a_[i]->get()) {
      widgets_[i]->set_alpha(gui_color_a_[i]->get());
      cmap_dirty_ = true;
    }

    if (widgets_[i]->get_shadeType() != gui_sstate_[i]->get()) {
      widgets_[i]->set_shadeType(gui_sstate_[i]->get());
      cmap_dirty_ = true;
    }
    
    if (widgets_[i]->get_onState() != gui_onstate_[i]->get()) {
      widgets_[i]->set_onState(gui_onstate_[i]->get());
      cmap_dirty_ = true;
    }
  }
}


void
EditColorMap2D::tcl_unpickle()
{
  widgets_.clear();

  gui_num_entries_.reset();
  resize_gui(gui_num_entries_.get());
  for (int i=0; i < gui_num_entries_.get(); i++)
  {
    gui_wstate_[i]->reset();
    if (gui_wstate_[i]->get()[0] == 't')
    {
      widgets_.push_back(scinew TriangleCM2Widget());
      widgets_[widgets_.size()-1]->tcl_unpickle(gui_wstate_[i]->get());
    }
    else if (gui_wstate_[i]->get()[0] == 'r')
    {
      widgets_.push_back(scinew RectangleCM2Widget());
      widgets_[widgets_.size()-1]->tcl_unpickle(gui_wstate_[i]->get());
    }
    else if (gui_wstate_[i]->get()[0] == 'i') {
      widgets_.push_back(scinew ImageCM2Widget());
      widgets_[widgets_.size()-1]->tcl_unpickle(gui_wstate_[i]->get());
    }
  }

  // Grab colors
  resize_gui();
  update_from_gui();
  redraw();
}


void
EditColorMap2D::select_widget(int widget, int object) {
  for (int i = 0; i < (int)widgets_.size(); i++)
  {
    widgets_[i]->unselect_all();
  }
  gui_selected_widget_.set(widget);
  gui_selected_object_.set(object);
  if (widget >= 0 && widget < (int)widgets_.size())
    widgets_[widget]->select(object);
}
  


void
EditColorMap2D::push(int x, int y, int button)
{
  button_ = button;
  first_motion_ = true;
  mouse_pick(x,y,button);
  select_widget(mouse_widget_, mouse_object_);
  redraw();
}

void
EditColorMap2D::mouse_pick(int x, int y, int b) {
  const bool right_button_down = (b==3);
  if (!right_button_down)
    for (mouse_widget_ = widgets_.size()-1; mouse_widget_>=0; mouse_widget_--)
      if (widgets_[mouse_widget_]->get_onState() &&
	  (mouse_object_ = widgets_[mouse_widget_]->pick1
	   (x, height_-1-y, width_, height_))) break;
  
  if (!mouse_object_)
    for (mouse_widget_ = widgets_.size()-1; mouse_widget_>=0; mouse_widget_--)
      if (widgets_[mouse_widget_]->get_onState() &&
	  (mouse_object_ = widgets_[mouse_widget_]->pick2
	   (x, height_-1-y, width_, height_, right_button_down))) break;
}

void
EditColorMap2D::set_window_cursor(int x, int y)
{
  const int old_mouse_wid = mouse_widget_;
  const int old_mouse_obj = mouse_object_;
  mouse_pick(x,y,0);
  if (old_mouse_wid != mouse_widget_ || old_mouse_obj != mouse_object_) {
    string cstr("crosshair");
    if (mouse_widget_ != -1) 
      cstr = widgets_[mouse_widget_]->tk_cursorname(mouse_object_);
    Tk_DefineCursor(ctx_->tkwin_, Tk_GetCursor
		    (the_interp, ctx_->tkwin_, ccast_unsafe(cstr)));
  }
}


void
EditColorMap2D::motion(int x, int y)
{
  if (button_ == 0) {
    set_window_cursor(x,y);
    return;
  }
  
  const int selected = gui_selected_widget_.get();
  if (selected != -1)
  {
    if (first_motion_)
    {
      undo_stack_.push(UndoItem(UndoItem::UNDO_CHANGE, selected,
				widgets_[selected]->clone()));
      first_motion_ = false;
    }

    widgets_[selected]->move(x, height_-1-y, width_, height_);
    cmap_dirty_ = true;
    updating_ = true;
    want_to_execute();
  }
}



void
EditColorMap2D::release(int x, int y)
{
  button_ = 0;
  set_window_cursor(x,y);
  const int selected = gui_selected_widget_.get();
  if (selected != -1)
  {
    widgets_[selected]->release(x, height_-1-y, width_, height_);
    updating_ = false;
    want_to_execute();
  }
}

void
EditColorMap2D::execute()
{
  ASSERTMSG(cmap_iport_, "EditColorMap2D: cant find input CM2 port");  
  ASSERTMSG(cmap_oport_, "EditColorMap2D: cant find output CM2 port");
  ASSERTMSG(hist_iport_, "EditColorMap2D: cant find input histo port");

  update_from_gui();
  ColorMap2Handle icmap = 0;
  cmap_iport_->get(icmap);
  
  if (icmap.get_rep() && icmap->generation != icmap_generation_) {
    widgets_ = icmap->widgets();
    icmap_generation_ = icmap->generation;
    cmap_dirty_ = true;
    if (!just_resend_selection_ && icmap->selected() != -1) 
      gui_selected_widget_.set(icmap->selected());
  }

  update_to_gui();

  NrrdDataHandle h = 0;
  hist_iport_->get(h);

  if (h.get_rep() && h->generation != hist_generation_) {
    hist_generation_ = h->generation;
    if(h->nrrd->dim != 2 && h->nrrd->dim != 3) {
      error("Invalid input dimension.");
    }
    histo_ = h->nrrd;
    histo_dirty_ = true;
  } else if (!h.get_rep()) {
    if(histo_ != 0)
      histo_dirty_ = true;
    histo_ = 0;
  }

  
  redraw();

  if (!just_resend_selection_)
    sent_cmap2_ = scinew ColorMap2(widgets_, updating_, gui_faux_.get());
  just_resend_selection_ = false;
  icmap_generation_ = sent_cmap2_->generation;
  sent_cmap2_->selected() = gui_selected_widget_.get();
  cmap_oport_->send(sent_cmap2_);

}


void EditColorMap2D::save_ppm(const string &filename,
			      int sx, int sy,
			      int bpp, const unsigned char * buf)
{
  int R = 3;
  int G = 2;
  int B = 1;

  if (isBigEndian()){
    R = 0;
    G = 1;
    B = 2;
  }
  
//  int A = 0;
  ofstream output(filename.c_str(), ios::out);
  if (!output) {
    error("ERROR: can't open file "+string(filename));
    return;
  }
  
  if ( bpp == 1 || bpp == 2 )
    output << "P2 \n# CREATOR: " << "\n"; // endl;
  else if ( bpp == 3 || bpp == 4 )
    output << "P6\n# CREATOR: " << "\n\n"; // endl;
  else {
    error("Error: unknown number of bytes per pixel " + to_string(bpp));
    return;
  }
  
  output << sx/4 << " " << sy/4 << "\n"; // endl;
  output << 255 << "\n"; // endl;
  
  for (int row = sy - 1; row >= 0; row-=4) {
    for (int col = 0; col < sx; col+=4) {
      int p = bpp * ( row * sx + col );
      switch (bpp) {
      case 2:
      case 1:
	output << (int) buf[p] << " \n"; // endl;
	break;
      default:
	output <<buf[p + R]<<buf[p + G]
	       <<buf[p + B];
	break;
      }
    }
  }
  if (output) output.close();
}


void
EditColorMap2D::init_shader_factory() 
{
  //----------------------------------------------------------------
  // decide what rasterization to use
  if (use_back_buffer_) {
    if(!shader_factory_) {
      shader_factory_ = new CM2ShaderFactory();

      if (sci_getenv("SCIRUN_SOFTWARE_CM2")) {
	if (use_back_buffer_) 
	  warning("SCIRUN_SOFWARE_CM2 set.  Rasterizing Colormap in software.");
	use_back_buffer_ = false;
	return;
      }

      FragmentProgramARB dummy("!!ARBfp1.0\nMOV result.color, 0.0;\nEND");
      if(dummy.create()) {
        dummy.destroy();
        use_back_buffer_ = false;
        remark ("Shaders not supported.");
	remark ("Using software rasterization for widgets.");
      } else {
        remark ("ARB Shaders are supported.");
	remark ("Using hardware rasterization for widgets.");
      }
      dummy.destroy();
    }
  }
}

void
EditColorMap2D::build_colormap_texture()
{
  glDrawBuffer(GL_BACK);
  glReadBuffer(GL_BACK);

  bool rebuild_texture = (!use_back_buffer_ && 
			  (width_ != colormap_texture_.dim2() || 
			   height_ != colormap_texture_.dim1()));

  if (!glIsTexture(colormap_texture_id_)) rebuild_texture = true;

  if (cmap_dirty_ || rebuild_texture) {
    // update texture
    if(rebuild_texture) {
      if(glIsTexture(colormap_texture_id_)) {
	glDeleteTextures(1, &colormap_texture_id_);
	colormap_texture_id_ = 0;
      }
      glGenTextures(1, &colormap_texture_id_);
    }
    
    glBindTexture(GL_TEXTURE_2D, colormap_texture_id_);

    if (rebuild_texture) {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
        
    if (use_back_buffer_) 
    {
      // Back Buffer /w Fragment shader rendering of colormap2 texture
      glEnable(GL_BLEND);
      glDrawBuffer(GL_BACK);
      glReadBuffer(GL_BACK);
      glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);  
      glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);  
      for(unsigned int i=0; i<widgets_.size(); i++) {
	if (! widgets_[i]->is_empty()) {
	  widgets_[i]->set_value_range(value_offset_, value_scale_);
	  widgets_[i]->rasterize(*shader_factory_, gui_faux_.get(), 0);
	}
      }
      if (rebuild_texture)
	glCopyTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8, 0,0,width_, height_,0);
      else
	glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0,width_, height_);
      glClearColor(0.0, 0.0, 0.0, 0.0);
      glClear(GL_COLOR_BUFFER_BIT);
    } else {
      // Software Rendering of Colormap Texture
      if (rebuild_texture) // realloc cmap
	colormap_texture_.resize(height_, width_, 4);
      colormap_texture_.initialize(0.0);
      for (unsigned int i=0; i<widgets_.size(); i++) {
	if (! widgets_[i]->is_empty()) {
	  widgets_[i]->set_value_range(value_offset_, value_scale_);
	  widgets_[i]->rasterize(colormap_texture_, gui_faux_.get());
	}
      }

      if (rebuild_texture)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 
		     colormap_texture_.dim2(), colormap_texture_.dim1(),
		     0, GL_RGBA, GL_FLOAT, &colormap_texture_(0,0,0));
      else
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 
			colormap_texture_.dim2(), colormap_texture_.dim1(),
			GL_RGBA, GL_FLOAT, &colormap_texture_(0,0,0));
    }
    
    glBindTexture(GL_TEXTURE_2D, 0);
    cmap_dirty_ = false;
  }
} 




void
EditColorMap2D::build_histogram_texture()
{
  if (!histo_dirty_) return;
  
  histo_dirty_ = false;
  value_offset_ = 0.0;
  value_scale_ = 1.0;
  
  if(glIsTexture(histogram_texture_id_)) {
    glDeleteTextures(1, &histogram_texture_id_);
    histogram_texture_id_ = 0;
  }
  
  glGenTextures(1, &histogram_texture_id_);
  glBindTexture(GL_TEXTURE_2D, histogram_texture_id_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  int axis_size[3];

  if (!histo_) {
    unsigned char zero = 0;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, 1,1,
		 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,  &zero);
    glBindTexture(GL_TEXTURE_2D, 0);
  } else {
    nrrdAxisInfoGet_nva(histo_, nrrdAxisInfoSize, axis_size);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, axis_size[histo_->dim-2],
		 axis_size[histo_->dim-1], 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,
		 histo_->data);
    glBindTexture(GL_TEXTURE_2D, 0);
    
    if (histo_->kvp) {
      const char *min = nrrdKeyValueGet(histo_, "jhisto_nrrd0_min");
      const char *max = nrrdKeyValueGet(histo_, "jhisto_nrrd0_max");
      const char *new_min = nrrdKeyValueGet(histo_, "jhisto_nrrd0_new_min");
      const char *new_max = nrrdKeyValueGet(histo_, "jhisto_nrrd0_new_max");
      double dmin, dmax, new_dmin, new_dmax;
      if (min && max && new_min && new_max && 
	  string_to_double(min, dmin) && string_to_double(max, dmax) &&
	  string_to_double(new_min, new_dmin) && string_to_double(new_max, new_dmax)) {
	value_offset_ = (new_dmin - dmin) / (dmax - dmin);
	value_scale_ = (new_dmax - new_dmin) / (dmax - dmin);
	cmap_dirty_ = true;
      }
    }
  }
}


void
EditColorMap2D::draw_texture(GLuint &texture_id)
{
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, texture_id);

  glBegin(GL_QUADS);
  {
    glTexCoord2f( 0.0,  0.0);
    glVertex2f( 0.0,  0.0);
    glTexCoord2f( 1.0,  0.0);
    glVertex2f( 1.0,  0.0);
    glTexCoord2f( 1.0,  1.0);
    glVertex2f( 1.0,  1.0);
    glTexCoord2f( 0.0,  1.0);
    glVertex2f( 0.0,  1.0);
  }
  glEnd();

  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_TEXTURE_2D);
}


void
EditColorMap2D::redraw()
{
  gui->lock();

  if(!ctx_ || ctx_->width() < 3 || ctx_->height() < 3 || !ctx_->make_current()) {
    gui->unlock(); 
    return; 
  }
  
  init_shader_factory();

  glDrawBuffer(GL_BACK);
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT);
  
  glViewport(0, 0, width_, height_);
  glViewport(0, 0, 512, 256);//width_, height_);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glScalef(2.0, 2.0, 2.0);
  glTranslatef(-0.5 , -0.5, 0.0);
  build_colormap_texture();
  glLoadIdentity();

  const float scale_factor = 2.0 * scale_.get();
  glScalef(scale_factor, scale_factor, scale_factor);
  glTranslatef(-0.5 - pan_x_.get(), -0.5 - pan_y_.get() , 0.0);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  

  // Draw Histogram
  build_histogram_texture();
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  glColor4f(gui_histo_.get(), gui_histo_.get(), gui_histo_.get(), 1.0);
  draw_texture(histogram_texture_id_);

  // Draw Colormap
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  draw_texture(colormap_texture_id_);

  glDisable(GL_BLEND);
  // Draw Colormap Widget Frames
  for(unsigned int i=0; i<widgets_.size(); i++) {
    if (! widgets_[i]->is_empty()) {
      widgets_[i]->draw();
    }
  }

  // draw outline of the image space.
  glColor4f(0.25, 0.35, 0.25, 1.0); 
  glBegin(GL_LINES);
  {
    glVertex2f( 0.0,  0.0);
    glVertex2f( 1.0,  0.0);

    glVertex2f( 1.0,  0.0);
    glVertex2f( 1.0,  1.0);

    glVertex2f( 1.0,  1.0);
    glVertex2f( 0.0,  1.0);

    glVertex2f( 0.0,  1.0);
    glVertex2f( 0.0,  0.0);
  }
  glEnd();

  if (save_ppm_) {
    unsigned int* FrameBuffer = scinew unsigned int[width_*height_];
    glFlush();
    glReadBuffer(GL_BACK);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    glReadPixels(0, 0,width_, height_,
		 GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, FrameBuffer);
    char tmp[256];
    sprintf(tmp, filename_.get().c_str());
    remark("Writing to file: "+filename_.get());
    save_ppm(tmp, width_, height_,
	     4, (const unsigned char *)(FrameBuffer));
    remark("Done saving framebuffer..."+filename_.get());
    delete FrameBuffer;
    save_ppm_ = false;
  }
  
  ctx_->swap();
  ctx_->release();
  gui->unlock();
}

} // end namespace SCIRun
