
#include <GL/glx.h>
#include <GL/glu.h>
#include <Packages/rtrt/visinfo/visinfo.h>
#include <Packages/rtrt/Core/remote.h>
#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
using namespace std;

#define PORT 4300

void serve(int sock);

struct WindowInfo {
  Display* dpy;
  int screen;
  Colormap cmap;
  XVisualInfo* vi;
  GLXContext cx;
  Window parentWindow;
  Window win;
};

int main()
{
  int listen_sock = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  if(setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0){
    perror("setsockopt");
    exit(1);
  }
  struct sockaddr_in sa;
  sa.sin_family = AF_INET;
  sa.sin_port = htons(PORT);
  sa.sin_addr.s_addr = INADDR_ANY;
  if(bind(listen_sock, (struct sockaddr*)&sa, sizeof(sa)) == -1){
    perror("bind");
    exit(1);
  }
  if(listen(listen_sock, 5) == -1){
    perror("listen");
    exit(1);
  }
  cerr << "remote server waiting for connections on port " << PORT << '\n';
  for(;;){
    struct sockaddr_in in_sa;
    socklen_t in_sa_len = sizeof(in_sa);
    int in = accept(listen_sock, (struct sockaddr*)&in_sa, &in_sa_len);
    if(in == -1){
      perror("accept");
      exit(1);
    }
    struct hostent* hent = gethostbyaddr(&in_sa.sin_addr, sizeof(in_sa.sin_addr),
					 AF_INET);
    if(!hent){
      unsigned long a = ntohl(in_sa.sin_addr.s_addr);
      cerr << "accepted connection from unknown host, ip addr=" << (a>>24) << "." << ((a>>16)&0xff) << "." << ((a>>8)&0xff) << "." << (a&0xff) << '\n';
    } else {
      cerr << "accepted connection from " << hent->h_name << '\n';
    }
    serve(in);
    if(!hent){
      unsigned long a = ntohl(in_sa.sin_addr.s_addr);
      cerr << "done with connection from unknown host, ip addr=" << (a>>24) << "." << ((a>>16)&0xff) << "." << ((a>>8)&0xff) << "." << (a&0xff) << '\n';
    } else {
      cerr << "done with connection from " << hent->h_name << '\n';
    }
  }
}

void createWindow(WindowInfo& winfo, Window parentWindow)
{
  winfo.parentWindow = parentWindow;
  XSetWindowAttributes atts;
  int flags=CWColormap|CWEventMask|CWBackPixmap|CWBorderPixel;
  atts.background_pixmap = None;
  atts.border_pixmap = None;
  atts.border_pixel = 0;
  atts.colormap=winfo.cmap;
  atts.event_mask=0;
  int xres=1, yres=1;
  winfo.win=XCreateWindow(winfo.dpy, parentWindow,
			  0, 0, xres, yres, 0, winfo.vi->depth,
			  InputOutput, winfo.vi->visual, flags, &atts);
  XMapWindow(winfo.dpy, winfo.win);
  
  winfo.cx=glXCreateContext(winfo.dpy, winfo.vi, NULL, True);
  if(!glXMakeCurrent(winfo.dpy, winfo.win, winfo.cx)){
    cerr << "glXMakeCurrent failed!\n";
  }
  glShadeModel(GL_FLAT);
  for(;;){
    XEvent e;
    XNextEvent(winfo.dpy, &e);
    if(e.type == MapNotify)
      break;
  }
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  glViewport(0, 0, xres, yres);
  glClearColor(0, 0, .2, 1);
  glClear(GL_COLOR_BUFFER_BIT);
  glXSwapBuffers(winfo.dpy, winfo.win);
  glFlush();
}

void resizeWindow(WindowInfo& winfo, int xres, int yres)
{
}

void setupStreams(WindowInfo& winfo)
{
}

void processMessage(WindowInfo& winfo, int sock)
{
  char buf[sizeof(RemoteMessage)];
  int total = 0;
  do {
    int n = read(sock, buf+total, sizeof(RemoteMessage)-total);
    if(n == -1){
      perror("read");
      exit(1);
    }
    total+=n;
  } while(total < (int)sizeof(RemoteMessage));
  RemoteMessage* rm = (RemoteMessage*)(&buf[0]);
  if(rm->len != sizeof(RemoteMessage)){
    cerr << "protocol error\n";
    exit(1);
  }
  RemoteReply reply;
  switch(rm->type){
  case RemoteMessage::CreateWindow:
    createWindow(winfo, rm->window);
    break;
  case RemoteMessage::ResizeWindow:
    resizeWindow(winfo, rm->xres, rm->yres);
    break;
  case RemoteMessage::SetupStreams:
    setupStreams(winfo);
    break;
  };
  reply.len = sizeof(reply);
  int nwrite = write(sock, &reply, sizeof(reply));
  if(nwrite != sizeof(reply)){
    cerr << "write error";
    exit(1);
  }
}

void serve(int inbound)
{
  WindowInfo winfo;
  // open window
  winfo.dpy=XOpenDisplay(NULL);
  if(!winfo.dpy){
    cerr << "Cannot open display\n";
    exit(1);
  }

  int error, event;
  if ( !glXQueryExtension( winfo.dpy, &error, &event) ) {
    cerr << "GL extension NOT available!\n";
    XCloseDisplay(winfo.dpy);
    winfo.dpy=0;
    exit(1);
  }
  winfo.screen=DefaultScreen(winfo.dpy);
  
  char* criteria="db, max rgb";
  if(!visPixelFormat(criteria)){
    cerr << "Error setting pixel format for visinfo\n";
    cerr << "Syntax error in criteria: " << criteria << '\n';
    exit(1);
  }
  int nvinfo;
  winfo.vi=visGetGLXVisualInfo(winfo.dpy, winfo.screen, &nvinfo);
  if(!winfo.vi || nvinfo == 0){
    cerr << "Error matching OpenGL Visual: " << criteria << '\n';
    exit(1);
  }
  winfo.cmap = XCreateColormap(winfo.dpy, RootWindow(winfo.dpy, winfo.screen),
			       winfo.vi->visual, AllocNone);

  fd_set fds;
  FD_ZERO(&fds);
  int* xfds;
  int numxfds;
  int maxfd=0;
  if(XInternalConnectionNumbers(winfo.dpy, &xfds, &numxfds) == 0){
    cerr << "XInternalConnectionNumbers failed\n";
    exit(1);
  }
  for(int i=0;i<numxfds;i++){
    if(xfds[i] > maxfd)
      maxfd=xfds[i];
    FD_SET(xfds[i], &fds);
  }
  if(inbound > maxfd)
    maxfd = inbound;
  FD_SET(inbound, &fds);

  XFlush(winfo.dpy);
  for(;;){
    fd_set readfds(fds);
    if(select(maxfd+1, &readfds, 0, 0, 0) == -1){
      perror("select");
      exit(1);
    }
    if(FD_ISSET(inbound, &fds)){
      cerr << "Need to read from remote\n";
    }
    bool dox=false;
    for(int i=0;i<numxfds;i++){
      if(FD_ISSET(xfds[i], &readfds)){
	XProcessInternalConnection(winfo.dpy, xfds[i]);
	dox=true;
      }
    }
    if(dox){
      while(XEventsQueued(winfo.dpy, QueuedAfterReading) > 0){
	XEvent e;
	XNextEvent(winfo.dpy, &e);	
	cerr << "event: " << e.type << '\n';
      }
    }
    if(FD_ISSET(inbound, &fds)){
      processMessage(winfo, inbound);
    }
  }
}
