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
#include <Core/GuiInterface/TclObj.h>
#include <Core/Containers/Array1.h>
#include <Core/2d/DrawObj.h>
#include <Core/2d/Widget.h>

namespace SCIRun {
  
  class Polyline;

class SCICORESHARE Diagram : public TclObj, public DrawObj {
private:
  Array1<bool> active_;
  Array1<Polyline *> poly_;
  Array1<Widget *> widget_;

  BBox2d graphs_bounds_;
  string window_;
  int selected_;
  int selected_widget_;
  
  typedef enum { SelectOne, SelectMany } SelectMode;
  typedef enum { ScaleAll, ScaleEach } ScaleMode;
  typedef enum { Draw, Pick } DrawMode;

  GuiInt *gui_select, *gui_scale;
/*   SelectMode select_mode; */
/*   ScaleMode scale_mode; */
  int select_mode_, scale_mode_;
  DrawMode draw_mode_;

public:
  Diagram( const string &name="" );
  virtual ~Diagram();

  void add( Polyline * );
  int add_widget( Widget *);
  void redraw() { if ( parent() ) parent()->need_redraw(); }
  virtual void reset_bbox();
  virtual void get_bounds(BBox2d&);

  virtual void tcl_command(TCLArgs&, void*);
  virtual void set_id( const string &);
  virtual void set_window( const string& );

  void get_active( Array1<Polyline *> &);

 private:  
  void button_press( int x, int y, int button );
  void button_motion( int x, int y, int button );
  void button_release( int x, int y, int button );

  void add_hairline();
  void add_zoom();

 public:
  // For OpenGL
#ifdef SCI_OPENGL
  virtual void draw();
#endif
  static PersistentTypeID type_id;
  
  virtual void io(Piostream&);    
  
};

void Pio(Piostream&, Diagram*&);

} // namespace SCIRun

#endif // SCI_Diagram_h
