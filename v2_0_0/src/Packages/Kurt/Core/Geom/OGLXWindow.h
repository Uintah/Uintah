#ifndef OGLXWINDOW_H
#define OGLXWINDOW_H

#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glu.h>
#include <vector>

using std::vector;

namespace Kurt {

class OGLXVisual;

class OGLXWindow {
public:
  OGLXWindow(char *name, char *dpyname,
	     OGLXVisual *v,
	     bool map   = false,
	     int width  = 640,
	     int height = 512,
	     int x      = 0,
	     int y      = 0);

  virtual ~OGLXWindow();
  
  virtual void init(){}
  virtual void handleEvent(){}

  void resize( int width, int height);
  void map();
  void unmap(); 
  

  void readFB(unsigned char* writebuffer,
	      int umin = 0, int vmin = 0,
	      int umax = 0, int vmax = 0);

  int width(){ return _width;}
  int height(){ return _height; }

protected:
  
  OGLXVisual *visual;
  char *name;
  char *dpyName;
  Display *dpy;
  GLXDrawable gwin;
  Window win;
  GLXContext cx;
  
private:
  bool _mapped;

protected:
  int _width, _height;
};
  
} // namespace Kurt
#endif
