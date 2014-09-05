#ifndef SCIREXWINDOW_H
#define SCIREXWINDOW_H

#include <Packages/Kurt/Core/Geom/OGLXWindow.h>
#include <Core/Thread/Runnable.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glu.h>
#include <vector>

using std::vector;


namespace SCIRun{
    class GeomObj;
    class Barrier;
    class Mutex;
    struct DrawInfoOpenGL;
    class Material;
}

using namespace SCIRun;
namespace Kurt {

class SCIRexRenderData;
class SCIRexWindow : public Runnable, public OGLXWindow {
public:

  SCIRexWindow(char *name, char *dpyname,
	       SCIRexRenderData *rd,
	       bool map   = false,
	       int width  = 640,
	       int height = 512,
	       int x      = 0,
	       int y      = 0);

  virtual ~SCIRexWindow();
  virtual void run();
  virtual void init();
  virtual void handleEvent();
  virtual void draw(DrawInfoOpenGL* di, Material* mat,
		    double time);
  void kill(){ die_ = true; }
  void addGeom(GeomObj *geom);
  unsigned char* getBuffer(){ return pbuffer; }

protected:
    vector<GeomObj*> geom_objs;
    
    virtual void setup();
    virtual int  eventmask();
    void resizeBuffers();
  void update_data();
    void reshape(int width, int height);
    virtual void draw();
    DrawInfoOpenGL *di;
    Material *mat;
    double time;
    const float *mv;
private:

  SCIRexRenderData *render_data_;
  GLubyte         *colorSave;
  GLfloat         *depthSave;
  unsigned char*  pbuffer;
  bool die_;
};
  
} //end namespace
#endif

