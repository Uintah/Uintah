#define GLUT_GLUI_THREAD 1

#include <Packages/rtrt/Core/DpyBase.h>
#include <Packages/rtrt/Core/Color.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Semaphore.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#ifdef GLUT_GLUI_THREAD
#  include <GL/glut.h>
#  if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#    pragma set woff 1430
#    pragma set woff 3201
#    pragma set woff 1375
#  endif
#  include <glui.h>
#  if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#    pragma reset woff 1430
#    pragma reset woff 3201
#    pragma reset woff 1375
#  endif
#endif

extern void run_gl_test();

using namespace rtrt;
using namespace SCIRun;
using namespace std;

class MyDpy;
class GGT;

class MyGui: public DpyBase {
public:
  MyGui():
    DpyBase("MyGui"),
#ifdef GLUT_GLUI_THREAD
    ggt(0), gg_dpy(0), gg_win(0),
#endif
    child(0), child_thread(0)
  {
  }

  void setChild(MyDpy* newchild) {
    child = newchild;
  }

  void setChildThread(Thread* newchild_thread) {
    child_thread = newchild_thread;
  }

#ifdef GLUT_GLUI_THREAD
  void setGlutGlui(GGT* new_ggt, Display* new_gg_dpy, Window new_gg_win) {
    ggt = new_ggt;
    gg_dpy = new_gg_dpy;
    gg_win = new_gg_win;
  }

protected:
  GGT* ggt;
  Display* gg_dpy;
  Window gg_win;
public:
#endif

protected:
  MyDpy* child;
  Thread* child_thread;

  virtual void run();
  virtual void init();
  virtual void display();
  virtual void key_pressed(unsigned long key);
};

//////////////////////////////////////////////////////////

// This class just draws as fast as it can.  It doesn't care about
// events.
class MyDpy: public DpyBase {
public:
  MyDpy(MyGui* parent):
    DpyBase("MyDpy"),
    parentSema("parent sema", 0)
  {
    parent->setChild(this);
  }

  Semaphore parentSema;

  void release(Window win) {
    parentWindow = win;
    parentSema.up();
  }
protected:
  
  Window parentWindow;
  double past;

  virtual void run() {
    parentSema.down();
    cerr << "MyDpy::run::parentSema down\n";
    // Open the window with the parent parameter.  Set receive events
    // to false.
    open_display(parentWindow, false);

    init();
  
    past = SCIRun::Time::currentSeconds();
    for(;;) {
      if (should_close()) {
        close_display();
        return;
      }
      
      display();
    }
  }
  
  virtual void display() {
    glShadeModel(GL_FLAT);
    //    glReadBuffer(GL_BACK);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 400, 0, 400);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.375, 0.375, 0.0);
    
    glClearColor(1,0,1, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    // draw stuff
    glColor3f(1,1,0);
    glBegin(GL_POLYGON);
    glVertex2f(50,50);
    glVertex2f(200,50);
    glVertex2f(200,200);
    glVertex2f(50,200);
    glEnd();
    
    double current=SCIRun::Time::currentSeconds();
    double framerate = 1./ (current - past);
    //cerr << "dt1 = " << (current - past) << ",\tcurrent = " << current << ",\tpast = " << past << endl;
    past = current;
    char buf[100];
    sprintf(buf, "%3.1ffps", framerate);
    printString(fontbase, 10, 3, buf, Color(1,1,1));
    
    glFinish();
    glXSwapBuffers(dpy, win);
    XFlush(dpy);

    glClearColor(1,0,1, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    // draw stuff
    glColor3f(0,1,1);
    glBegin(GL_POLYGON);
    glVertex2f(100,100);
    glVertex2f(300,100);
    glVertex2f(300,300);
    glVertex2f(100,300);
    glEnd();
    
    //    for (int i = 0; i < 1e8; i++);
    current=SCIRun::Time::currentSeconds();
    framerate = 1./ (current - past);
    //cerr << "dt2 = " << (current - past) << ",\tcurrent = " << current << ",\tpast = " << past << endl;
    past = current;
    sprintf(buf, "%3.1ffps", framerate);
    printString(fontbase, 10, 3, buf, Color(1,1,1));
    
    glFinish();
    glXSwapBuffers(dpy, win);
    XFlush(dpy);
  }
};

///////////////////////////////////////////////////

#ifdef GLUT_GLUI_THREAD

// This needs to be global to satisfy glut
static GGT* activeGui;
extern "C" Display *__glutDisplay;
extern "C" Window** __glutWindowList;

class GGT: public Runnable {
public:
  GGT():
    mainWindowId(-1)
  {
    activeGui = this;
  }

  virtual ~GGT() {
    GLUI_Master.close_all();
    cerr << "GLUI_Master.close_all finished\n";
    glutDestroyWindow(activeGui->mainWindowId);
    cerr << "glutDestroyWindow finished\n";
  }
  
  virtual void run() {
    printf("before glutInit\n");
    char *argv = "GGT Thread run";
    int argc = 1;
    glutInit( &argc, &argv );
    printf("after glutInit\n");
    
    // Initialize GLUT and GLUI stuff.
    printf("start glut inits\n");
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(1, 1 );
    glutInitWindowPosition( 100, 0 );
    
    mainWindowId = glutCreateWindow("GG Controls");

    dpy = __glutDisplay;
    win = __glutWindowList[mainWindowId-1][1];
    cerr << "initial win = "<<win<<"\n";

    // Setup callback functions
    glutDisplayFunc( GGT::display );

    // Must do this after glut is initialized.
    createMenus();

    printf("end glut inits\n");
    
    glutMainLoop();
  }

  static void close(int external) {
    if (external) {
      // Generate a signal to the window to close
      XEvent event;
      event.type = KeyPress;
      event.xkey.keycode = XK_q;
      event.xkey.x = 0;
      event.xkey.y = -1;
      event.xkey.window = activeGui->win;
      cerr << "external close generating an event\n";
      XSendEvent(activeGui->dpy, activeGui->win, false, 0, &event);
    } else {
      // This is internal, just shutdown the thread
      Thread::exit();
    }
  }

  static void display(void) {
    glutSetWindow(activeGui->mainWindowId);
    glClearColor(1,1,0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    cerr << "GGT::display called\n";
  }

  static void keyboard(unsigned char key, int x, int y) {
    cerr << "keyboard called with key '"<<key<<"' at ("<<x<<", "<<y<<")\n";
    switch (key) {
    case 'q':
      cerr << "'q' received\n";
      close(0);
      break;
    }
  }

  static void reshape( int width, int height ) {
    static bool first=true;
    if(first){
      activeGui->win = __glutWindowList[activeGui->mainWindowId-1][1];
      cerr << "winid=" << activeGui->win << '\n';
      first=false;
    }
  }
protected:
  void createMenus() {
    // Register call backs with the glui controls.
    GLUI_Master.set_glutKeyboardFunc( GGT::keyboard );
    GLUI_Master.set_glutReshapeFunc( GGT::reshape );
    GLUI_Master.set_glutIdleFunc( NULL );

    // Create the sub window.
    GLUI* glui_subwin =
      GLUI_Master.create_glui_subwindow( mainWindowId, GLUI_SUBWINDOW_RIGHT);

    glui_subwin->set_main_gfx_window( mainWindowId );

    GLUI_Panel * main_panel   = glui_subwin->add_panel( "" );
    glui_subwin->add_button_to_panel(main_panel, "Exit Thread", 0, GGT::close);
    
  }

public:
  int mainWindowId;
  Display *dpy;
  Window win;
};

#endif // ifdef GLUT_GLUI_THREAD

///////////////////////////////////////////////////

void MyGui::run() {
  cerr << "MyGui::run(): start\n";
  open_events_display();
  cerr << "MyGui::run(): after open_events_display\n";
  
  init();
  cerr << "MyGui::run(): after init\n";
  
  for(;;) {
    if (should_close()) {
      close_display();
      return;
    }

    if (redraw) {
      redraw = false;
      display();
    }
    
    wait_and_handle_events();
  }
}

void MyGui::init() {
  child->release(win);
  cerr << "MyGui::init::parentSema up\n";
}
  
void MyGui::key_pressed(unsigned long key) {
  switch (key) {
#ifdef GLUT_GLUI_THREAD
  case XK_g:
    ggt->close(1);
    break;
#endif
  case XK_r:
    post_redraw();
    break;
  case XK_c:
    child->stop();
    break;
  case XK_p:
    stop();
    break;
  case XK_q:
#ifdef GLUT_GLUI_THREAD
    // Close the GG thread
    ggt->close(1);
#endif
    // Close the children
    //    Thread::exitAll(0);
    child->stop();
    //      close_display();
    stop();
    //Thread::exit();
    break;
  case XK_Escape:
    Thread::exitAll(0);
    break;
  case XK_s:
    child_thread->stop();
    break;
  }
}

void MyGui::display() {
  cerr << "MyGui::display called\n";
}

///////////////////////////////////////////////////

int main(int argc, char *argv[]) {

  if (argc < 2)
    run_gl_test();

#if 1
  // Create a new GUI thread
  MyGui* gui = new MyGui();
  // Make a new Display
  Thread* dpythread = new Thread(new MyDpy(gui), "Dpy");
  dpythread->detach();
  gui->setChildThread(dpythread);
  (new Thread(gui, "Gui"))->detach();
#endif

#ifdef GLUT_GLUI_THREAD
  if (strcmp(argv[1], "glut") == 0) {
    // start up the glut glui thread
    GGT* gg_runner = new GGT();
#if 0
    gg_runner->run();
#else
    Thread* gg_thread = new Thread(gg_runner, "GG Thread");
    gg_thread->detach();
    gui->setGlutGlui(gg_runner,0,0);
#endif
  }
#endif
  
  return 0;
}
