/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rightsget_iports(name Reserved.
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

class SCICORESHARE Diagram : public DrawGui {
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
