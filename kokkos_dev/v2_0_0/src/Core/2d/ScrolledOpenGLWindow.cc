/*
 *  ScrolledOpenGLWindow.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   Aug 20001
 *
 */


#include <GL/gl.h>
#include <sci_glu.h>
#include <GL/glx.h>

#include <tcl.h>
#include <tk.h>

#include <Core/2d/ScrolledOpenGLWindow.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


ScrolledOpenGLWindow::ScrolledOpenGLWindow(GuiInterface* gui)
  : OpenGLWindow(gui)
{
}
 

void
ScrolledOpenGLWindow::tcl_command(GuiArgs& args, void* userdata)
{
  OpenGLWindow::tcl_command( args, userdata );
}


} // End namespace SCIRun


