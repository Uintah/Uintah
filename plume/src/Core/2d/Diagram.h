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
 *  Diagram.h: Displayable 2D object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_Diagram_h
#define SCI_Diagram_h 

#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/Array1.h>
#include <Core/2d/DrawGui.h>
#include <Core/2d/DrawObj.h>
#include <Core/2d/Widget.h>
#include <stack>

namespace SCIRun {
  using std::stack;
  
class ScrolledOpenGLWindow;

class Diagram : public DrawGui {
private:
  ScrolledOpenGLWindow *ogl_;

  Array1<bool> active_;
  Array1<DrawObj *> poly_;
  Array1<Widget *> widget_;

  stack<int> zoom_stack_;
  BBox2d graphs_bounds_;
  string window_;
  int selected_;
  int selected_widget_;
  
  typedef enum { SelectOne, SelectMany } SelectMode;
  typedef enum { ScaleAll, ScaleEach } ScaleMode;
  typedef enum { Draw, Pick } DrawMode;
  typedef enum { NormalMode, ZoomInMode, ZoomOutMode } OperateMode;

  GuiInt *gui_select, *gui_scale;
/*   SelectMode select_mode; */
/*   ScaleMode scale_mode; */
  int select_mode_, scale_mode_;
  DrawMode draw_mode_;
  OperateMode operate_mode_;
  bool changed_;
  GuiContext* ctx;

public:
  Diagram(GuiInterface* gui, const string &name="" );
  virtual ~Diagram();

  void add( DrawObj * );
  int add_widget( Widget *);
  void redraw();
  void update() { if (parent() ) parent()->need_redraw(); }
  virtual void reset_bbox();
  virtual void get_bounds(BBox2d&);

  virtual void tcl_command(GuiArgs&, void*);
  virtual void set_id( const string &);
  virtual void set_windows( const string &menu, const string &tb,
			    const string &ui, const string &ogl);

  void get_active( Array1<DrawObj *> &);
  //! the following two convert from [0-1] to world space coordinates
  double x_get_at( double );
  double y_get_at( double );
 private:  
  void button_press( int x, int y, int button );
  void button_motion( int x, int y, int button );
  void button_release( int x, int y, int button );

  void pan_start( int x, int y, int button );
  void pan_move( int x, int y, int button );
  void pan_end( int x, int y, int button );

  void add_hairline();
  void add_axes();
  void add_zoom();

  void zoom_in( int x, int y, int );
  void zoom_out( int x, int y, int );

  void child_changed( DrawObj *) { changed_ = true; }
 public:
  // For OpenGL
#ifdef SCI_OPENGL
  virtual void draw( bool = false );
#endif
  static PersistentTypeID type_id;
  
  virtual void io(Piostream&);    
  
};

void Pio(Piostream&, Diagram*&);

} // namespace SCIRun

#endif // SCI_Diagram_h
