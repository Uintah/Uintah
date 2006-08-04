
#include <Packages/Kurt/Core/Geom/SCIRexWindow.h>
#include <Packages/Kurt/Core/Geom/OGLXVisual.h>
#include <Packages/Kurt/Core/Geom/SCIRexRenderData.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <strstream>
#include <fstream>
using std::ifstream;
using std::ofstream;
using std::ostrstream;

using std::cerr;
using std::endl;


namespace Kurt {

using SCIRun::GeomObj;
using SCIRun::Material;
using SCIRun::Thread;
using SCIRun::Barrier;



SCIRexWindow::~SCIRexWindow()
{
//   glXDestroyWindow( dpy, win ); // GLX 1.3
  XDestroyWindow(dpy, win);
}
SCIRexWindow::SCIRexWindow(char *name, char *dpyname,
			   SCIRexRenderData *rd, bool map,
			   int width, int height, int x, int y) :
  OGLXWindow( name, dpyname, rd->visual_, map, width, height, x, y), 
  render_data_(rd), colorSave(0), depthSave(0), pbuffer(0),
  die_(false)
{
 NOT_FINISHED("SCIRexWindow::SCIRexWindow");
}

void 
SCIRexWindow::run()
{
  render_data_->mutex_->lock(); 
  init();
  render_data_->mutex_->unlock();
  for(;;){
    while (XPending(dpy)){ 
      handleEvent();
    } 
    for(;;){
      int i = 0;
//       render_data_->mutex_->lock();     
//       cerr<<"check for Exit "<<my_thread_->getThreadName()<<"   "<<i++<<endl;
//       render_data_->mutex_->unlock();
      render_data_->barrier_->wait(render_data_->waiters_);
      if( die_ ){  
// 	render_data_->mutex_->lock(); 
// 	cerr<<"Returning from thread "<<my_thread_->getThreadName()<<endl;
// 	render_data_->mutex_->unlock();
	unmap();
	return;
      } else if( !render_data_->waiters_changed_ ){
	break;
      }
    }

//     render_data_->mutex_->lock(); 
//     cerr<<"update info for "<<my_thread_->getThreadName()<<endl;
//     render_data_->mutex_->unlock();
    
    update_data();
    
    render_data_->barrier_->wait( render_data_->waiters_);
//     render_data_->mutex_->lock(); 
//     cerr<<"render windows "<<my_thread_->getThreadName()<<endl;
//     render_data_->mutex_->unlock();
    
//   render_data_->mutex_->lock(); 
    draw();
//   render_data_->mutex_->unlock();

    render_data_->barrier_->wait( render_data_->waiters_);
//     render_data_->mutex_->lock(); 
//     cerr<<"wait on compositers "<<my_thread_->getThreadName()<<endl;
//     render_data_->mutex_->unlock();
    render_data_->barrier_->wait( render_data_->waiters_);
//     render_data_->mutex_->lock(); 
//     cerr<<"wait on Display "<<my_thread_->getThreadName()<<endl;
//     render_data_->mutex_->unlock();
    render_data_->barrier_->wait( render_data_->waiters_);
  }
}


void SCIRexWindow::init()
{
//   int major, minor;
//   const char *string_data;
  dpy = XOpenDisplay(dpyName);
  if (!dpy) { // error(name, "can't open display");
    cerr<<name<<" can't open display: "<<dpyName<<"!\n";
    return;
  }

//   if (glXQueryVersion(dpy, &major, &minor)) {
//     if (major == 1) {
//       if (minor < 3) // error(name, "need GLX 1.3");
// 	cerr<<name<<" need GLX 1.3\n";
//       else {
// 	string_data = glXQueryServerString(dpy, DefaultScreen(dpy),
// 					   GLX_VERSION);
// 	cerr<< string_data<<endl;
// 	if (strncmp(string_data,"1.3", 3) == 0) {
// 	  cerr<<name<<" got GLX 1.3\n";
// 	setup();
// 	} else
// 	  //error(name, "need GLX 1.3");
// 	  cerr<<name<<" need GLX 1.3\n";
//       }
//     }
//   }
  setup();

//   for(;;){
//     XEvent e;
//     XNextEvent(dpy, &e);
//     if(e.type == MapNotify)
//       break;
//   }
}

void 
SCIRexWindow::addGeom(GeomObj *geom)
{
  geom_objs.push_back( geom );
}

int SCIRexWindow::eventmask()
{
   return  ExposureMask | StructureNotifyMask;
}
  
static Bool WaitForNotify(Display *d, XEvent *e, char *arg) {
   return (e->type == MapNotify) && (e->xmap.window == (Window)arg);
}

void SCIRexWindow::setup()
{
  GLXFBConfig *fbc = NULL;  // GLX 1.3
  //GLXFBConfigSGIX *fbc;  // pseudo GLX 1.3
  XVisualInfo *vi;
   Colormap cmap;
   XSetWindowAttributes swa;
   //   Window win;
   XEvent event;
   int nelements;

//    /* GLX 1.3 */
//    /* Find a FBConfig that uses RGBA.  Note that no attribute list is */
//    /* needed since GLX_RGBA_BIT is a default attribute.               */
   fbc = glXChooseFBConfig(dpy, DefaultScreen(dpy),
			   visual->attributes(), &nelements); //1.3

//       fbc = glXChooseFBConfigSGIX(dpy, DefaultScreen(dpy), 0, &nelements);
   vi = glXGetVisualFromFBConfig(dpy, *fbc); // 1.3
//       vi =  glXGetVisualFromFBConfigSGIX(dpy, fbc[0]); // pseudo 1.3
//    if (!vi) // error(name, "no suitable visual");
//      cerr<<name<<" no suitable visual\n";
 
//    /* Create a GLX context using the first FBConfig in the list. */
//    cx = glXCreateNewContext(dpy, fbc[0], GLX_RGBA_TYPE, 0, GL_FALSE);//1.3
//       cx = glXCreateContextWithConfigSGIX(dpy, fbc[0], // pseudo 1.3
// 					  GLX_RGBA_TYPE, 
// 					  0, GL_FALSE);
// *************************** GLX 1.2 ***********************************
//    int attributeList[] = { GLX_DOUBLEBUFFER, False,
//                            GLX_RED_SIZE, 4,
//                            GLX_GREEN_SIZE, 4,
//                            GLX_BLUE_SIZE, 4,
//                            GLX_ALPHA_SIZE, 4,
//                            GLX_TRANSPARENT_TYPE, GLX_TRANSPARENT_RGB,
//                            None};

//      int alist[] = { GLX_RGBA, None };
//    vi = glXChooseVisual(dpy, DefaultScreen(dpy), alist );
//     vi = glXChooseVisual(dpy, DefaultScreen(dpy), visual->attributes() );
//     if (!vi){
//       /* error(name, "no suitable visual"); */
//       cerr<<name<<" no suitable visual\n";
//     }
    cx = glXCreateContext(dpy, vi, 0, GL_TRUE);
// ***********************************************************************
   /* Create a colormap */
   cmap = XCreateColormap(dpy, RootWindow(dpy, vi->screen),vi->visual, AllocNone);
 
   /* Create a window */
   swa.colormap = cmap;
   swa.border_pixel = 0;
   swa.event_mask = this->eventmask();
   win = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 0, 0, _width,
		       _height, 0, vi->depth, InputOutput,
                       vi->visual, CWBorderPixel|CWColormap|CWEventMask, &swa);
   XMapWindow(dpy, win);
   XStoreName(dpy, win, name);
   XIfEvent(dpy, &event, WaitForNotify, (char*)win);
 
   /* Create a GLX window using the same FBConfig that we used for the */
   /* the GLX context.                                                 */
//    this->win = glXCreateWindow(dpy, fbc[0], win, 0);
 
   /* Connect the context to the window for read and write */
//        glXMakeContextCurrent(dpy, this->win, this->win, cx);
   glXMakeCurrent(dpy, win, cx);
}


void 
SCIRexWindow::resizeBuffers(void)
{
    if(colorSave != NULL)
        free(colorSave);
    colorSave = (GLubyte *)malloc(_width * _height * 4 * sizeof(GLubyte));
    if(depthSave != NULL)
        free(depthSave);
    depthSave = (GLfloat *)malloc(_width * _height * sizeof(GLfloat));
}

void 
SCIRexWindow::reshape(int width, int height)
{
    glViewport(0, 0, width, height);
    _width = width;
    _height = height;
    resizeBuffers();
}


void SCIRexWindow::handleEvent()
{

  bool redraw = false;
  
  do{
    XEvent event;
    XNextEvent(dpy, &event);
    switch(event.type) {

    case Expose:
      redraw = true;
      break;
    case ConfigureNotify:
//       glViewport(0, 0, event.xconfigure.width, event.xconfigure.height);
      redraw = true;
      break;
    case ResizeRequest:
      cerr<<"ResizeRequest ";
//       reshape( event.xresizerequest.width, event.xresizerequest.height);
      redraw = true;
      break;
    default:
      break;
    }
  } while (XPending(dpy));

//   if(redraw)
//      draw();
}

void SCIRexWindow::update_data()
{
  SCIRexRenderData *rd = render_data_;
  if( rd->viewport_changed_){
    resize( rd->viewport_x_, rd->viewport_y_);
    if (pbuffer != 0)
      delete [] pbuffer;
    pbuffer = new unsigned char[ 4 * rd->viewport_x_ * rd->viewport_y_];
  }
}

void
SCIRexWindow::draw(DrawInfoOpenGL* di, Material* mat, double time)
{
  this->di = di;
  this->mat = mat;
  this->time = time;
//   this->mv = mv;
  draw();
}

void 
SCIRexWindow::draw(){
  SCIRexRenderData *rd = render_data_;
  if( cx != glXGetCurrentContext() ){
//     glXMakeContextCurrent(dpy, win, win, cx);
    glXMakeCurrent(dpy, win, cx);
  }

  glDrawBuffer(GL_FRONT);
  //  glClearColor(0.2,0.2,0.5,1.0);
  glShadeModel(GL_SMOOTH);
  glMatrixMode( GL_PROJECTION );
  if(rd->pmat_)
    glLoadMatrixd( rd->pmat_);

  glMatrixMode( GL_MODELVIEW );
  //glPushMatrix();
  if(rd->mvmat_)
    glLoadMatrixd( rd->mvmat_ );


  if(rd->use_depth_){
    cerr<<"Using depth\n";
    glClear(GL_DEPTH_BUFFER_BIT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glEnable(GL_DEPTH_TEST);  
    glDrawPixels(rd->viewport_x_,
		 rd->viewport_y_,
		 GL_DEPTH_COMPONENT,
		 GL_UNSIGNED_BYTE, rd->depth_buffer_);
    
    glClearColor(0,0,0,0.0);
    glClear(GL_COLOR_BUFFER_BIT);
    
    
    
    glDepthMask(GL_FALSE);
    vector<GeomObj *>::iterator i;
    for( i = geom_objs.begin(); i != geom_objs.end(); i++){
      (*i)->draw( rd->di_, rd->mat_, rd->time_);
    }
    glDepthMask(GL_TRUE);
    glDisable(GL_DEPTH_TEST);  //glPopMatrix();
  } else {
    cerr<<"Not using depth\n";
    glClearColor(0,0,0,0.0);
    glClear(GL_COLOR_BUFFER_BIT);
    vector<GeomObj *>::iterator i;
    for( i = geom_objs.begin(); i != geom_objs.end(); i++){
      (*i)->draw( rd->di_, rd->mat_, rd->time_);
    }
  }    
  readFB(pbuffer, 0,0,_width, _height);
  
  if(rd->dump_){
    ostrstream convert;
    convert.width(4);
    convert.fill('0');
    convert << rd->curFrame_;
    char number[10];
    sprintf(number, "%04d\0",  rd->curFrame_);
    cerr<<"current frame is "<<rd->curFrame_<<endl;
    string n(name);
    n = n+number+".raw";
    int size = 3*_width*_height;
    ofstream dumpfile(n.c_str());
    dumpfile.write((const char *)pbuffer,size);
  }
}


} // namespace Kurt
