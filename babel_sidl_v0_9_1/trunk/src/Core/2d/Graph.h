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
 *  Graph.h:
 *
 *  Written by:
 *   Yarden Livnat
 *   July 20001
 *
 */


#ifndef Graph_h
#define Graph_h

#include <sgi_stl_warnings_off.h>
#include <string>
#include <map>
#include <sgi_stl_warnings_on.h>

#include <sci_gl.h>

#include <Core/Thread/ConditionVariable.h>
#include <Core/2d/DrawObj.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/Array1.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TclObj.h>
#include <Core/2d/DrawGui.h>
#include <Core/Thread/Mutex.h>
#include <Core/Datatypes/Color.h>
#include <Core/2d/OpenGLWindow.h>



namespace SCIRun {

class GraphHelper;

class Graph : public DrawGui {
  friend class SCIRun::GraphHelper;
private:
  DrawGui *obj_;
  GraphHelper *helper_;
  ConditionVariable has_work_;

public:
  Graph(GuiInterface* gui, const string & );
  virtual ~Graph() {}

  void lock();
  void unlock();

  void add( const string &, DrawGui *);
  virtual void tcl_command(GuiArgs&, void*);
  virtual void set_window( const string &);
  void update();

  virtual void need_redraw();
  virtual void get_bounds( BBox2d &) {}
  virtual void draw( bool = false ) {}
  virtual void io(Piostream& stream);

};


} // End namespace SCIRun

#endif /* Graph_h */


