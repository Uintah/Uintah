/*
 *  ScrolledOpenGLWindow.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   Aug 20001
 *
 */


#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>

#include <tcl.h>
#include <tk.h>

#include <Core/2d/ScrolledOpenGLWindow.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/TCL.h>

namespace SCIRun {


ScrolledOpenGLWindow::ScrolledOpenGLWindow()
  : OpenGLWindow()
{
}
 

void
ScrolledOpenGLWindow::tcl_command(TCLArgs& args, void* userdata)
{
  OpenGLWindow::tcl_command( args, userdata );
}


} // End namespace SCIRun


