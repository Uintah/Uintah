

#include <Packages/rtrt/Core/DpyBase.h>
#include <Packages/rtrt/Core/Color.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Semaphore.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

extern void run_gl_test();

using namespace rtrt;
using namespace SCIRun;
using namespace std;

class MyDpy;

class MyGui: public DpyBase {
public:
  MyGui():
    DpyBase("MyGui"),
    child(0), child_thread(0)
  {
  }

  void setChild(MyDpy* newchild) {
    child = newchild;
  }

  void setChildThread(Thread* newchild_thread) {
    child_thread = newchild_thread;
  }
protected:
  MyDpy* child;
  Thread* child_thread;

  virtual void run();
  virtual void init();
  virtual void display();
  virtual void key_pressed(unsigned long key);
};

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

int main(int argc, char *argv[]) {

  if (argc < 2)
    run_gl_test();

  // Create a new GUI thread
  MyGui* gui = new MyGui();
  // Make a new Display
  Thread* dpythread = new Thread(new MyDpy(gui), "Dpy");
  dpythread->detach();
  gui->setChildThread(dpythread);
  (new Thread(gui, "Gui"))->detach();
  

  return 0;
}
