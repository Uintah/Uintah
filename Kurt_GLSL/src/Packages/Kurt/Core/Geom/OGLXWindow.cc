

#include <Packages/Kurt/Core/Geom/OGLXWindow.h>
#include <Packages/Kurt/Core/Geom/OGLXVisual.h>


namespace Kurt {

OGLXWindow::~OGLXWindow(){ glXDestroyWindow( dpy, win );}

OGLXWindow::OGLXWindow(char *name, char *dpyname,
		       OGLXVisual *v, bool map,
		       int width, int height, int x, int y) :
  name(name), dpyName(dpyname), visual(v),_mapped(map),
  _width(width), _height(height)
{
  // not done
}
  


void
OGLXWindow::resize(int width, int height)
{
  _width  = width;
  _height = height;
  XResizeWindow(dpy, win, _width, _height);
  XFlush(dpy);
  glViewport(0, 0, _width, _height);

}

void 
OGLXWindow::map(void)
{
  XMapWindow(dpy, win);
  XFlush(dpy);
}


void 
OGLXWindow::unmap(void)
{
  XUnmapWindow(dpy, win);
  XFlush(dpy);
}


void 
OGLXWindow::readFB(unsigned char* writebuffer,
       int umin, int vmin,
       int umax, int vmax)
{
//   if( cx != glXGetCurrentContext() )
//     glXMakeContextCurrent(dpy, win, win, cx);
//     glXMakeCurrent(dpy, win, cx);
  glFlush(); // Make sure the pipeline is clear.    
  if (umax == 0 || umax > (_width - 1))
    umax = _width - 1;
  if (vmax == 0 || vmax > (_height -1))
    vmax = _height - 1;
  if (umin > umax)
    umin = umax;
  if (vmin > vmax)
    vmin = vmax;    
  glPixelStorei(GL_PACK_ALIGNMENT,1);
  glReadBuffer(GL_FRONT);
  glReadPixels(umin, vmin, umax-umin+1, vmax-vmin+1,
	       GL_RGBA, GL_UNSIGNED_BYTE, writebuffer);
}


} // end namespace Kurt
