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
//  
//    File   : EditTransferFunc2.cc
//    Author : Milan Ikits
//    Author : Michael Callahan
//    Date   : Thu Jul  8 01:50:58 2004

#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Array3.h>

#include <sci_gl.h>
#include <Packages/Volume/Core/Util/Pbuffer.h>
#include <Packages/Volume/Core/Util/ShaderProgramARB.h>
#include <Packages/Volume/Dataflow/Ports/Colormap2Port.h>
#include <Packages/Volume/Core/Datatypes/CM2Shader.h>
#include <Packages/Volume/Core/Datatypes/CM2Widget.h>
#include <Dataflow/Ports/NrrdPort.h>

#include <Core/Geom/GeomOpenGL.h>
#include <tcl.h>
#include <tk.h>
#include <stack>
#include <iostream>

// tcl interpreter corresponding to this module
extern Tcl_Interp* the_interp;

// the OpenGL context structure
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);

using namespace SCIRun;
using namespace SCITeem;
using std::stack;

namespace Volume {


struct UndoItem
{
  enum Action { UNDO_CHANGE, UNDO_ADD, UNDO_DELETE };

  int action_;
  int selected_;
  CM2Widget *widget_;

  UndoItem(int a, int s, CM2Widget *w)
    : action_(a), selected_(s), widget_(w)
  {
  }

};


class EditTransferFunc2 : public Module {

  GLXContext ctx_;
  Display* dpy_;
  Window win_;
  int width_, height_;
  bool button_;
  vector<CM2Widget*> widgets_;
  stack<UndoItem> undo_stack_;
  CM2ShaderFactory* shader_factory_;
  Pbuffer* pbuffer_;
  Array3<float> array_;
  bool use_pbuffer_;
  bool use_back_buffer_;
  
  Nrrd* histo_;
  bool histo_dirty_;
  GLuint histo_tex_;

  bool cmap_dirty_;
  bool cmap_size_dirty_;
  GLuint cmap_tex_;
  
  int pick_widget_; // Which widget is selected.
  int pick_object_; // The part of the widget that is selected.
  bool first_motion_; // Push on undo when motion occurs, not on select.

  bool updating_; // updating the tf or not

  GuiInt gui_faux_;
  GuiDouble gui_histo_;

  GuiInt			gui_num_entries_;
  vector<GuiString *>		gui_name_;
  vector<GuiDouble *>		gui_color_r_;
  vector<GuiDouble *>		gui_color_g_;
  vector<GuiDouble *>		gui_color_b_;
  vector<GuiDouble *>		gui_color_a_;
  vector<GuiString *>           gui_wstate_;

public:
  EditTransferFunc2(GuiContext* ctx);
  virtual ~EditTransferFunc2();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  virtual void presave();
  
  void resize_gui(int n = -1);
  void update_from_gui();
  void update_to_gui(bool forward = true);
  void tcl_unpickle();

  void undo();

  void redraw();

  void push(int x, int y, int button, int modifier);
  void motion(int x, int y);
  void release(int x, int y, int button);
};



DECLARE_MAKER(EditTransferFunc2)

EditTransferFunc2::EditTransferFunc2(GuiContext* ctx)
  : Module("EditTransferFunc2", ctx, Filter, "Visualization", "Volume"),
    ctx_(0), dpy_(0), win_(0), button_(0), shader_factory_(0),
    pbuffer_(0), use_pbuffer_(true), use_back_buffer_(true),
    histo_(0), histo_dirty_(false), histo_tex_(0),
    cmap_dirty_(true), cmap_tex_(0),
    pick_widget_(-1), pick_object_(0), first_motion_(true), updating_(false),
    gui_faux_(ctx->subVar("faux")),
    gui_histo_(ctx->subVar("histo")),
    gui_num_entries_(ctx->subVar("num-entries"))
{
  widgets_.push_back(scinew TriangleCM2Widget());
  widgets_.push_back(scinew RectangleCM2Widget());
  resize_gui(2);
  update_to_gui(false);
}


EditTransferFunc2::~EditTransferFunc2()
{
  delete pbuffer_;
  delete shader_factory_;
}


void
EditTransferFunc2::tcl_command(GuiArgs& args, void* userdata)
{
  if (args.count() < 2) {
    args.error("No command for EditTransferFunc");
    return;
  }

  // mouse commands are motion, down, release
  if (args[1] == "mouse") {
    //cerr << "EVENT: mouse" << endl;
    int x, y, b, m;
    string_to_int(args[3], x);
    string_to_int(args[4], y);
    if (args[2] == "motion") {
      if (button_ == 0) // not buttons down!
	return;
      motion(x, y);
    } else {
      string_to_int(args[5], b); // which button it was
      if (args[2] == "push") {
	string_to_int(args[6], m); // which button it was
	push(x, y, b, m);
      } else {
	release(x, y, b);
      }
    }
  } else if (args[1] == "resize") {
    //cerr << "EVENT: resize" << endl;
    string_to_int(args[2], width_);
    string_to_int(args[3], height_);
    redraw();
  } else if (args[1] == "expose") {
    //cerr << "EVENT: expose" << endl;
    redraw();
  } else if (args[1] == "redraw") {
    //cerr << "EVENT: expose" << endl;
    reset_vars();
    if (args.count() > 2)
    {
      cmap_dirty_ = true;
    }
    redraw();
  } else if (args[1] == "redrawcmap") {
    //cerr << "EVENT: expose" << endl;
    cmap_dirty_ = true;
    redraw();
  } else if (args[1] == "closewindow") {
    //cerr << "EVENT: close" << endl;
    ctx_ = 0;
  } else if (!args[1].compare(0, 13, "color_change-")) {
    int n;
    string_to_int(args[1].substr(13), n);

    gui_num_entries_.reset();
    resize_gui();
    gui_color_r_[n]->reset();
    gui_color_g_[n]->reset();
    gui_color_b_[n]->reset();
    gui_color_a_[n]->reset();
    const double r = gui_color_r_[n]->get();
    const double g = gui_color_g_[n]->get();
    const double b = gui_color_b_[n]->get();
    const double a = gui_color_a_[n]->get();
    Color c(widgets_[n]->color());
    if (r != c.r() || g != c.g() || b != c.b() || a != widgets_[n]->alpha())
    {
      undo_stack_.push(UndoItem(UndoItem::UNDO_CHANGE, n,
				widgets_[n]->clone()));
      update_from_gui();
      want_to_execute();
    }
  } else if (args[1] == "unpickle") {
    tcl_unpickle();
  } else if (args[1] == "addtriangle") {
    widgets_.push_back(scinew TriangleCM2Widget());
    undo_stack_.push(UndoItem(UndoItem::UNDO_ADD, widgets_.size()-1, NULL));
    cmap_dirty_ = true;
    redraw();
    update_to_gui();
    want_to_execute();
  } else if (args[1] == "addrectangle") {
    widgets_.push_back(scinew RectangleCM2Widget());
    undo_stack_.push(UndoItem(UndoItem::UNDO_ADD, widgets_.size()-1, NULL));
    cmap_dirty_ = true;
    redraw();
    update_to_gui();
    want_to_execute();
  } else if (args[1] == "deletewidget") {
    if (pick_widget_ != -1 && pick_widget_ < (int)widgets_.size())
    {
      // Delete widget.
      undo_stack_.push(UndoItem(UndoItem::UNDO_DELETE,
				pick_widget_, widgets_[pick_widget_]));
      widgets_.erase(widgets_.begin() + pick_widget_);
      pick_widget_ = -1;
      cmap_dirty_ = true;
      redraw();
      update_to_gui();
      want_to_execute();
    }
  } else if (args[1] == "undowidget") {
    undo();
  } else if (args[1] == "reset_gui") {
    
  } else {
    Module::tcl_command(args, userdata);
  }
}


void
EditTransferFunc2::presave()
{
  unsigned int i;

  resize_gui();
  update_to_gui(false);

  // Pickle up the tcl states.
  for (i = 0; i < widgets_.size(); i++)
  {
    gui_wstate_[i]->set(widgets_[i]->tcl_pickle());
  }

  // Delete all of the unused variables.
  for (i = widgets_.size(); i < gui_name_.size(); i++)
  {
    const string num = to_string(i);
    ctx->erase("name-" + num);
    ctx->erase(num +"-color-r");
    ctx->erase(num +"-color-g");
    ctx->erase(num +"-color-b");
    ctx->erase(num +"-color-a");
    ctx->erase("state-" + num);

    delete gui_name_[i];
    delete gui_color_r_[i];
    delete gui_color_g_[i];
    delete gui_color_b_[i];
    delete gui_color_a_[i];
    delete gui_wstate_[i];
  }
  if (widgets_.size() < gui_name_.size())
  {
    const unsigned int ws = widgets_.size();
    gui_name_.erase(gui_name_.begin() + ws, gui_name_.end());
    gui_color_r_.erase(gui_color_r_.begin() + ws, gui_color_r_.end());
    gui_color_g_.erase(gui_color_g_.begin() + ws, gui_color_g_.end());
    gui_color_b_.erase(gui_color_b_.begin() + ws, gui_color_b_.end());
    gui_color_a_.erase(gui_color_a_.begin() + ws, gui_color_a_.end());
    gui_wstate_.erase(gui_wstate_.begin() + ws, gui_wstate_.end());
  }
}


void
EditTransferFunc2::undo()
{
  bool gui_update = false;
  if (!undo_stack_.empty())
  {
    const UndoItem &item = undo_stack_.top();
    
    switch (item.action_)
    {
    case UndoItem::UNDO_CHANGE:
      delete widgets_[item.selected_];
      widgets_[item.selected_] = item.widget_;
      gui_update = true;
      break;

    case UndoItem::UNDO_ADD:
      delete widgets_[item.selected_];
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
    redraw();
    want_to_execute();
    if (gui_update)
    {
      update_to_gui();
    }
  }
}


void
EditTransferFunc2::resize_gui(int n)
{
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
  }

  if (i != 0)
  {
    ctx->erase("marker");
    GuiString marker(ctx->subVar("marker"));
    marker.set("end");
  }
}


void
EditTransferFunc2::update_to_gui(bool forward)
{
  // Update GUI
  resize_gui();
  for (unsigned int i = 0; i < widgets_.size(); i++)
  {
    gui_name_[i]->set("Generic");
    Color c(widgets_[i]->color());
    gui_color_r_[i]->set(c.r());
    gui_color_g_[i]->set(c.g());
    gui_color_b_[i]->set(c.b());
    gui_color_a_[i]->set(widgets_[i]->alpha());
  }
  if (forward) { gui->execute(id + " create_entries"); }
}


void
EditTransferFunc2::update_from_gui()
{
  gui_num_entries_.reset();
  resize_gui();
  for (unsigned int i = 0; i < widgets_.size(); i++)
  {
    gui_color_r_[i]->reset();
    gui_color_g_[i]->reset();
    gui_color_b_[i]->reset();
    gui_color_a_[i]->reset();
    widgets_[i]->set_color(Color(gui_color_r_[i]->get(),
				 gui_color_g_[i]->get(),
				 gui_color_b_[i]->get()));
    widgets_[i]->set_alpha(gui_color_a_[i]->get());
  }
  cmap_dirty_ = true;
  redraw();
}


void
EditTransferFunc2::tcl_unpickle()
{
  unsigned int i;
  for (i=0; i <widgets_.size(); i++)
  {
    delete widgets_[i];
  }
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
    else
    {
      cerr << "EditTransferFunc debug: No widget state saved.\n";
    }
  }

  // Grab colors
  resize_gui();
  update_from_gui();
}


void
EditTransferFunc2::push(int x, int y, int button, int modifier)
{
  //cerr << "push: " << x << " " << y << " " << button << " " << width_ << " " << height_ << " " << modifier << endl;

  int i;

  button_ = button;
  first_motion_ = true;

  for (i = 0; i < (int)widgets_.size(); i++)
  {
    widgets_[i]->unselect_all();
  }

  pick_widget_ = -1;
  pick_object_ = 0;
  if (modifier == 0)
  {
    for (i = widgets_.size()-1; i >= 0; i--)
    {
      const int tmp = widgets_[i]->pick1(x, height_-1-y, width_, height_);
      if (tmp)
      {
	pick_widget_ = i;
	pick_object_ = tmp;
	widgets_[i]->select(tmp);
	break;
      }
    }
  }
  if (pick_widget_ == -1)
  {
    for (i = widgets_.size()-1; i >= 0; i--)
    {
      const int m = modifier;
      const int tmp = widgets_[i]->pick2(x, height_-1-y, width_, height_, m);
      if (tmp)
      {
	pick_widget_ = i;
	pick_object_ = tmp;
	widgets_[i]->select(tmp);
	break;
      }
    }
  }
  if (pick_widget_ != -1)
  {
    redraw();
  }
}



void
EditTransferFunc2::motion(int x, int y)
{
  //cerr << "motion: " << x << " " << y << endl;

  if (pick_widget_ != -1)
  {
    if (first_motion_)
    {
      undo_stack_.push(UndoItem(UndoItem::UNDO_CHANGE, pick_widget_,
				widgets_[pick_widget_]->clone()));
      first_motion_ = false;
    }

    widgets_[pick_widget_]->move(pick_object_, x, height_-1-y, width_, height_);
    cmap_dirty_ = true;
    updating_ = true;
    update_to_gui(true);
    redraw();
    want_to_execute();
  }
}



void
EditTransferFunc2::release(int x, int y, int button)
{
  //cerr << "release: " << x << " " << y << " " << button << endl;

  button_ = 0;
  if (pick_widget_ != -1)
  {
    widgets_[pick_widget_]->release(pick_object_, x, height_-1-y, width_, height_);
    updating_ = false;
    cmap_dirty_ = true;

    redraw();
    update_to_gui(true);
    want_to_execute();
  }
}



void
EditTransferFunc2::execute()
{
  //cerr << "EditTransferFunc2::execute" << endl;
  
  NrrdIPort* histo_port = (NrrdIPort*)get_iport("Histogram");
  if(histo_port) {
    NrrdDataHandle h;
    histo_port->get(h);
    if(h.get_rep()) {
      if(h->nrrd->dim != 2 && h->nrrd->dim != 3) {
        error("Invalid input dimension.");
      }
      if(histo_ != h->nrrd) {
        histo_ = h->nrrd;
        histo_dirty_ = true;
      }
    } else {
      if(histo_ != 0)
        histo_dirty_ = true;
      histo_ = 0;
    }
  } else {
    if(histo_ != 0)
      histo_dirty_ = true;
    histo_ = 0;
  }

  if(histo_dirty_) {
    redraw();
  }
  
  Colormap2OPort* cmap_port = (Colormap2OPort*)get_oport("Output Colormap");
  if (cmap_port) {
    Colormap2Handle cmap(scinew Colormap2(widgets_, updating_, gui_faux_.get()));
    cmap_port->send(cmap);
  }
}

void
EditTransferFunc2::redraw()
{
  gui->lock();

  //----------------------------------------------------------------
  // obtain rendering ctx 
  if(!ctx_) {
    const string myname(".ui" + id + ".f.gl.gl");
    Tk_Window tkwin = Tk_NameToWindow(the_interp, ccast_unsafe(myname),
                                      Tk_MainWindow(the_interp));
    if(!tkwin) {
      warning("Unable to locate window!");
      gui->unlock();
      return;
    }
    dpy_ = Tk_Display(tkwin);
    win_ = Tk_WindowId(tkwin);
    ctx_ = OpenGLGetContext(the_interp, ccast_unsafe(myname));
    width_ = Tk_Width(tkwin);
    height_ = Tk_Height(tkwin);
    // check if it was created
    if(!ctx_) {
      error("Unable to obtain OpenGL context!");
      gui->unlock();
      return;
    }
    glXMakeCurrent(dpy_, win_, ctx_);
#ifdef HAVE_GLEW
    sci_glew_init();
#endif
  } else {
    glXMakeCurrent(dpy_, win_, ctx_);
  }
  
  //----------------------------------------------------------------
  // decide what rasterization to use
  if(use_pbuffer_ || use_back_buffer_) {
    if(!shader_factory_) {
      shader_factory_ = new CM2ShaderFactory();
      FragmentProgramARB dummy("!!ARBfp1.0\nMOV result.color, 0.0;\nEND");
      if(dummy.create()) {
        dummy.destroy();
        use_pbuffer_ = false;
        use_back_buffer_ = false;
        cerr << "[EditTransferFunction2] Shaders not supported; "
             << "switching to software rasterization" << endl;
      }
      dummy.destroy();
    }
  }

  if(use_pbuffer_ && (!pbuffer_ || pbuffer_->width() != width_
                      || pbuffer_->height() != height_)) {
    if(pbuffer_) {
      pbuffer_->destroy();
      delete pbuffer_;
    }
    pbuffer_ = new Pbuffer(width_, height_, GL_INT, 8, true, GL_FALSE);
    if(pbuffer_->create()) {
      pbuffer_->destroy();
      delete pbuffer_;
      pbuffer_ = 0;
      use_pbuffer_ = false;
      cerr << "[EditTransferFunction2] Pbuffers not supported; "
           << "switching to back buffer rasterization" << endl;
    } else {
      use_back_buffer_ = false;
      cerr << "[EditTransferFunction2] Using Pbuffer rasterization" << endl;
    }
  }

  //----------------------------------------------------------------
  // update local array
  if(use_pbuffer_) {
    // pbuffer rasterization
    if(cmap_dirty_) {
      pbuffer_->makeCurrent();
      glDrawBuffer(GL_FRONT);
      glViewport(0, 0, width_, height_);

      glClearColor(0.0, 0.0, 0.0, 0.0);
      glClear(GL_COLOR_BUFFER_BIT);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      glTranslatef(-1.0, -1.0, 0.0);
      glScalef(2.0, 2.0, 2.0);

      glDisable(GL_DEPTH_TEST);
      glDisable(GL_LIGHTING);
      glDisable(GL_CULL_FACE);

      glEnable(GL_BLEND);
      glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    
      // Rasterize widgets
      for (unsigned int i=0; i<widgets_.size(); i++) {
        widgets_[i]->rasterize(*shader_factory_, gui_faux_.get(), 0);
      }

      glDisable(GL_BLEND);
      pbuffer_->swapBuffers();
      glXMakeCurrent(dpy_, win_, ctx_);
      cmap_dirty_ = false;
    }
  } else {
    // software rasterization
    bool cmap_size_dirty = width_ != array_.dim2() || height_ != array_.dim1();
    if(cmap_dirty_ || cmap_size_dirty) {
      // realloc cmap
      if(cmap_size_dirty)
        array_.resize(height_, width_, 4);
      // clear cmap
      for(int i=0; i<array_.dim1(); i++) {
        for(int j=0; j<array_.dim2(); j++) {
          array_(i,j,0) = 0.0;
          array_(i,j,1) = 0.0;
          array_(i,j,2) = 0.0;
          array_(i,j,3) = 0.0;
        }
      }
      // rasterize widgets
      for (unsigned int i=0; i<widgets_.size(); i++) {
	widgets_[i]->rasterize(array_, gui_faux_.get());
      }
      // update texture
      if(cmap_size_dirty) {
        if(glIsTexture(cmap_tex_)) {
          glDeleteTextures(1, &cmap_tex_);
          cmap_tex_ = 0;
        }
        glGenTextures(1, &cmap_tex_);
        glBindTexture(GL_TEXTURE_2D, cmap_tex_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, array_.dim2(), array_.dim1(),
                     0, GL_RGBA, GL_FLOAT, &array_(0,0,0));
        glBindTexture(GL_TEXTURE_2D, 0);
      } else {
        glBindTexture(GL_TEXTURE_2D, cmap_tex_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, array_.dim2(), array_.dim1(),
                        GL_RGBA, GL_FLOAT, &array_(0,0,0));
        glBindTexture(GL_TEXTURE_2D, 0);
      }
      cmap_dirty_ = false;
    }
  }

  //----------------------------------------------------------------
  // update histo tex
  if(histo_dirty_) {
    if(glIsTexture(histo_tex_)) {
      glDeleteTextures(1, &histo_tex_);
      histo_tex_ = 0;
    }
    glGenTextures(1, &histo_tex_);
    glBindTexture(GL_TEXTURE_2D, histo_tex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    int axis_size[3];
    nrrdAxisInfoGet_nva(histo_, nrrdAxisInfoSize, axis_size);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, axis_size[histo_->dim-2],
                 axis_size[histo_->dim-1], 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,
                 histo_->data);
    glBindTexture(GL_TEXTURE_2D, 0);
    histo_dirty_ = false;
  }
  
  //----------------------------------------------------------------
  // draw
  glDrawBuffer(GL_BACK);
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT);
  
  glViewport(0, 0, width_, height_);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(-1.0, -1.0, 0.0);
  glScalef(2.0, 2.0, 2.0);

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glDisable(GL_CULL_FACE);

  // draw cmap
  if(use_back_buffer_) {
    // rasterize widgets into back buffer
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    for(unsigned int i=0; i<widgets_.size(); i++) {
      widgets_[i]->rasterize(*shader_factory_, gui_faux_.get(), 0);
    }

    glBlendFunc(GL_ONE, GL_DST_ALPHA);
    // draw histo
    if(histo_) {
      glActiveTexture(GL_TEXTURE0);
      glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, histo_tex_);
      double alpha = gui_histo_.get();
      glColor4f(alpha, alpha, alpha, 1.0);
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
    } else {
      glColor4f(0.0, 0.0, 0.0, 0.0);
      glBegin(GL_QUADS);
      {
        glVertex2f( 0.0,  0.0);
        glVertex2f( 1.0,  0.0);
        glVertex2f( 1.0,  1.0);
        glVertex2f( 0.0,  1.0);
      }
      glEnd();
    }
    glDisable(GL_BLEND);
  } else {
    // draw histo
    if(histo_) {
      glActiveTexture(GL_TEXTURE0);
      glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, histo_tex_);
      double alpha = gui_histo_.get();
      glColor4f(alpha, alpha, alpha, 1.0);
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
    // draw textures
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    if(use_pbuffer_) {
      pbuffer_->bind(GL_FRONT);
    } else {
      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, cmap_tex_);
    }
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
    if(use_pbuffer_) {
      pbuffer_->release(GL_FRONT);
    } else {
      glBindTexture(GL_TEXTURE_2D, 0);
      glDisable(GL_TEXTURE_2D);
    }
    glDisable(GL_BLEND);
  }
  
  // draw widgets
  for(unsigned int i=0; i<widgets_.size(); i++) {
    widgets_[i]->draw();
  }
  
  glXSwapBuffers(dpy_, win_);
  glXMakeCurrent(dpy_, 0, 0);

  gui->unlock();
}


} // end namespace Volume
