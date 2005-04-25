#define GLUT_GLUI_THREAD 1

#include <Packages/rtrt/Core/DpyBase.h>
#include <Packages/rtrt/Core/Color.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Thread/Semaphore.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <unistd.h> // For sleep()

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
    ggt(0),
#endif
    child(0), child_thread(0)
  {
    cleaned = true;
  }

  void setChild(MyDpy* newchild) {
    child = newchild;
  }

  void setChildThread(Thread* newchild_thread) {
    child_thread = newchild_thread;
  }

#ifdef GLUT_GLUI_THREAD
  void setGlutGlui(GGT* new_ggt) {
    ggt = new_ggt;
  }

  void startGGT();
  
protected:
  GGT* ggt;
public:
#endif

  virtual void cleanup();

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
  MyDpy():
    DpyBase("MyDpy", DoubleBuffered, false),
    parentSema("parent sema", 0)
  {
    dont_close();
  }

  Semaphore parentSema;

  void release(Window win) {
    parentWindow = win;
    parentSema.up();
  }

  void wait_on_close() {
    parentSema.down();
    // Now wait for the thread to have exited
    unsigned int i =0;
    while(my_thread_ != 0) {
      i++;
      if (i %10000 == 0)
        cerr << "+";
    }
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
        if (!glXMakeCurrent(dpy,None, NULL)) {
          cerr << "glXMakeCurrent failed\n";
        } else {
          cerr << "glXMakeCurrent succeeded\n";
        }
        cleanup();
        parentSema.up();
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

    //    usleep(1000);
    
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

    //    usleep(1000);
  }
};

///////////////////////////////////////////////////

#ifdef GLUT_GLUI_THREAD

// This needs to be global to satisfy glut
static GGT* activeGui = NULL;
//extern "C" Display *__glutDisplay;
//extern "C" Window** __glutWindowList;

class GGT: public Runnable {
public:
  GGT():
    gui(0),
    opened(false),
    on_death_row(false),
    mainWindowId(-1)
  {
    activeGui = this;
  }

  void setGui(MyGui* new_gui) {
    gui = new_gui;
  }
  
  virtual ~GGT() {
    cerr << "GGT::~GGT() called\n";
    cleanup();
  }

  void stop() {
    on_death_row = true;
  }
  
  void cleanup() {
    if (!opened) return;
    else opened = false;
    
    DpyBase::xlock();
    GLUI_Master.close_all();
    cerr << "GLUI_Master.close_all finished\n";
    glutDestroyWindow(mainWindowId);
    cerr << "glutDestroyWindow finished\n";
    //    XCloseDisplay(dpy);
    //    cerr << "XCloseDisplay for GGT finished\n";
    DpyBase::xunlock();
  }
  
  virtual void run() {
    DpyBase::xlock();

    printf("before glutInit\n");
    char *argv = "GGT Thread run";
    int argc = 1;
    glutInit( &argc, &argv );
    printf("after glutInit\n");
    
    // Initialize GLUT and GLUI stuff.
    printf("start glut inits\n");
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(135, 70 );
    glutInitWindowPosition( 100, 0 );
    
    mainWindowId = glutCreateWindow("GG Controls");

//     dpy = __glutDisplay;
//     win = __glutWindowList[mainWindowId-1][1];
//     cerr << "initial win = "<<win<<"\n";

    // Setup callback functions
    glutDisplayFunc( GGT::display );

    // Must do this after glut is initialized.
    createMenus();

    opened = true;
    
    DpyBase::xunlock();
    printf("end glut inits\n");
    
    glutMainLoop();
  }

  static void close(int mode) {
    cerr << "GGT::close("<<mode<<")\n";
    switch (mode) {
    case 0:
      // We are just exiting ourselves
      
      // Tell the Gui that we've left.
      if (activeGui->gui) activeGui->gui->setGlutGlui(0);
      // This is internal, just shutdown the thread.
      Thread::exit();
      break;
    case 2:
      // Tell the GUI thread to go bye bye.  It would be nice if this
      // worked.  For now call Thread::exitAll();
      //      if (activeGui->gui) activeGui->gui->stop();
      Thread::exitAll(0);
      break;
    }
  }

  static void display(void) {
    glutSetWindow(activeGui->mainWindowId);
    glClearColor(1,1,0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    cerr << "GGT::display called\n";
  }

  static void keyboard(unsigned char key, int x, int y) {
    //    cerr << "keyboard called with key '"<<key<<"' at ("<<x<<", "<<y<<")\n";
    switch (key) {
    case 'q':
      //      cerr << "'q' received\n";
      close(0);
      break;
    }
  }

  static void reshape( int width, int height ) {
    static bool first=true;
    if(first){
//       activeGui->win = __glutWindowList[activeGui->mainWindowId-1][1];
//       cerr << "winid=" << activeGui->win << '\n';
      first=false;
    }
  }

  static void idle(void) {
    // Check to see if we need to go bye bye
    if (activeGui->on_death_row)
      close(0);
    else
      usleep(1000);
  }
protected:
  void createMenus() {
    // Register call backs with the glui controls.
    GLUI_Master.set_glutKeyboardFunc( GGT::keyboard );
    GLUI_Master.set_glutReshapeFunc( GGT::reshape );
    GLUI_Master.set_glutIdleFunc( GGT::idle );

    // Create the sub window.
    GLUI* glui_subwin =
      GLUI_Master.create_glui_subwindow( mainWindowId, GLUI_SUBWINDOW_RIGHT);

    glui_subwin->set_main_gfx_window( mainWindowId );

    GLUI_Panel * main_panel   = glui_subwin->add_panel( "" );
    glui_subwin->add_button_to_panel(main_panel, "Exit Thread", 0, GGT::close);
    glui_subwin->add_button_to_panel(main_panel, "Exit All", 2, GGT::close);
  }

  MyGui* gui;
  bool opened;
  bool on_death_row;
public:
  int mainWindowId;
  Display *dpy;
  Window win;
};

void MyGui::startGGT() {
  if (ggt || activeGui) return; // already have one
  setGlutGlui(new GGT());
  ggt->setGui(this);
  Thread* gg_thread = new Thread(ggt, "GG Thread");
  gg_thread->detach();
}

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
      cleanup();
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
    if (shift_pressed) startGGT();
    else if (ggt) ggt->stop();
    break;
#endif
  case XK_r:
    redraw = true;
    break;
//   case XK_c:
//     child->stop();
//     break;
//   case XK_p:
//     stop();
//     break;
  case XK_q:
    cleaned = false;
    stop();
    break;
  case XK_Escape:
    Thread::exitAll(0);
    break;
  case XK_s:
    child_thread->stop();
    break;
  }
}

void MyGui::cleanup() {
  if (cleaned) return;
  else cleaned = true;
  
#ifdef GLUT_GLUI_THREAD
  // Close the GG thread
  if (ggt) ggt->stop();
#endif
  // Close the children
  child->stop();

  // Wait for the child to stop rendering
  child->wait_on_close();
  // Can't delete it for now, because it will cause a recursive lock
  // when doing Thread::exitAll().

  //  delete(child);

  close_display();
}

void MyGui::display() {
  cerr << "MyGui::display called\n";
}

///////////////////////////////////////////////////

int main(int argc, char *argv[]) {

  if (argc < 2) {
    run_gl_test();
    exit(0);
  }

  //  DpyBase::initUseXThreads();

  ThreadGroup *main_group = new ThreadGroup("main group");
  
#if 1
  // Create the Dpy first
  MyDpy* dpy = new MyDpy();
  // Create a new GUI thread
  MyGui* gui = new MyGui();
  // Set up the relationship
  gui->setChild(dpy);
  // Make a new Display
  Thread* dpythread = new Thread(dpy, "Dpy", main_group);
  gui->setChildThread(dpythread);
  new Thread(gui, "Gui", main_group);
#endif

#ifdef GLUT_GLUI_THREAD
  if (strcmp(argv[1], "glut") == 0) {
    // start up the glut glui thread
    gui->startGGT();
  }
#endif

  main_group->join();

  return 0;
}
