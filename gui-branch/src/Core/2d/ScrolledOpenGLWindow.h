
/*
 *  ScrolledOpenGLWindow.h:
 *
 *  Written by:
 *   Yarden Livnat
 *   July 20001
 *
 */


#ifndef ScrolledOpenGLWindow_h
#define ScrolledOpenGLWindow_h

#include <string>
#include <map>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>

#include <Core/2d/OpenGLWindow.h>


namespace SCIRun {

class ScrolledOpenGLWindow : public OpenGLWindow {
private:
  
public:
  ScrolledOpenGLWindow();
  virtual ~ScrolledOpenGLWindow() {}

  virtual void tcl_command(TCLArgs&, void*);
  
};



} // End namespace SCIRun

#endif ScrolledOpenGLWindow_h


