
#include <GL/glx.h>
#include <GL/glu.h>
#include <Packages/rtrt/visinfo/visinfo.h>
#include <Packages/rtrt/Core/remote.h>
#include <Packages/rtrt/Core/Image.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>
#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <vector>
#include <strings.h>
#include <errno.h>
using namespace std;
using namespace rtrt;
using namespace SCIRun;
#define MAXFRAMES 2

static Mutex io("io lock");

void serve(int sock);

namespace rtrt {
  class WindowInfo;
  class RStream : public Runnable {
  public:
    RStream(WindowInfo& winfo, int listen_sock, int myidx);
    ~RStream();
    volatile bool shutdown;
    virtual void run();
  private:
    char buf[MAXBUFSIZE];
    WindowInfo& winfo;
    int listen_sock;
    int myidx;
  };

  struct WindowInfo {
    Display* dpy;
    int screen;
    Colormap cmap;
    XVisualInfo* vi;
    GLXContext cx;
    Window parentWindow;
    Window win;
    vector<RStream*> streams;
    vector<Thread*> streamthreads;
    bool done;
    int listen_sock;
    int sendPipe;
    int recvPipe;
    Mutex lock;
    Image* images[MAXFRAMES];
    char pad0[128];
    volatile int curFrame;
    char pad1[128];
    volatile int pendingCurFrame;
    char pad2[128];
    int curFrames[MAXSTREAMS];
    char pad3[128];
    WindowInfo() : lock("WindowInfo sync lock") {
      pendingCurFrame = -1;
      curFrame = 0;
      for(int i=0;i<MAXSTREAMS;i++)
	curFrames[i]=0;
    }
  };
}

int main()
{
  int listen_sock = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  if(setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0){
    perror("setsockopt");
    Thread::exitAll(1);
  }
  struct sockaddr_in sa;
  sa.sin_family = AF_INET;
  sa.sin_port = htons(PORT);
  sa.sin_addr.s_addr = INADDR_ANY;
  if(bind(listen_sock, (struct sockaddr*)&sa, sizeof(sa)) == -1){
    perror("bind");
    Thread::exitAll(1);
  }
  if(listen(listen_sock, 5) == -1){
    perror("listen");
    Thread::exitAll(1);
  }
  for(;;){
    cerr << "remote server waiting for connections on port " << PORT << '\n';
    struct sockaddr_in in_sa;
#ifdef __linux
    socklen_t in_sa_len = sizeof(in_sa);
#else
    int in_sa_len = sizeof(in_sa);
#endif
    int in = accept(listen_sock, (struct sockaddr*)&in_sa, &in_sa_len);
    if(in == -1){
      perror("accept");
      Thread::exitAll(1);
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
  atts.event_mask=StructureNotifyMask;
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
}

void resizeWindow(WindowInfo& winfo, int xres, int yres)
{
  XResizeWindow(winfo.dpy, winfo.win, xres, yres);
  XFlush(winfo.dpy);
  for(int i=0;i<MAXFRAMES;i++)
    winfo.images[i] = new Image(xres, yres, false);
}

RStream::RStream(WindowInfo& winfo, int listen_sock, int myidx)
  : winfo(winfo), listen_sock(listen_sock), myidx(myidx)
{
  shutdown=false;
}

RStream::~RStream()
{
}


void RStream::run()
{
  int sock = accept(listen_sock, 0, 0);
  cerr << "Stream " << this << " started\n";
  if(sock == -1){
    perror("accept");
    Thread::exitAll(1);
  }
  io.lock();
  cerr << "Receive thread accepted connection...\n";
  io.unlock();
  for(;;){
    int total = 0;
    do {
      long n = read(sock, buf+total, sizeof(int)-total);
      if(n == 0 || shutdown){
	close(sock);
	return;
      }
      if(n == -1){
	perror("1. read");
	Thread::exitAll(1);
      }
      total += n;
    } while(total < sizeof(int));
    int len = *(int*)buf;
    if(len > MAXBUFSIZE){
      cerr << "Protocol error, len=" << len << '\n';
      Thread::exitAll(1);
    }
    do {
      long n = read(sock, buf+total, len-total);
      if(n == 0 || shutdown){
	close(sock);
	return;
      }
      if(n == -1){
	perror("2. read");
	Thread::exitAll(1);
      }
      total += n;
    } while(total < len);
    
    int idx = sizeof(int);
    int framenumber = *(int*)(&buf[idx]);
    idx += sizeof(int);
    int curFrame = winfo.curFrame;
    if(framenumber < curFrame){
      io.lock();
      cerr << "Received data for old frame: " << framenumber << " while on frame " << curFrame << '\n';
      io.unlock();
      continue;
    } else if(framenumber == curFrame){
      // Normal, just unpack below...
    } else {
      // update my curFrames and posssibly sync
      if(winfo.curFrames[myidx] < framenumber){
	winfo.curFrames[myidx] = framenumber;
	int i;
	int pendingCurFrame = winfo.pendingCurFrame;
	for(i=0;i<winfo.streams.size();i++)
	  if(winfo.curFrames[i] <= pendingCurFrame)
	    break;
	if(i == winfo.streams.size()){
	  int writecount=0;
	  winfo.lock.lock();
	  if(winfo.pendingCurFrame != framenumber){
	    writecount = framenumber-winfo.pendingCurFrame;
	    winfo.pendingCurFrame = framenumber;
	  }
	  winfo.lock.unlock();
	  for(int i=0;i<writecount;i++){
	    int b;
	    long s = write(winfo.sendPipe, &b, sizeof(b));
	    if(s != sizeof(b)){
	      perror("write");
	      Thread::exitAll(1);
	    }
	  }
	}
	if(framenumber >= curFrame+MAXFRAMES){
	  io.lock();
	  cerr << "Stream reader " << myidx << " is too far ahead, spinning...\n";
	  cerr << "pendingCurFrame=" << winfo.pendingCurFrame << ", curFrame=" << winfo.curFrame << "frameno=" << framenumber << '\n';
	  io.unlock();
	  while(framenumber > winfo.curFrame+1 && !shutdown) {}
	  io.lock();
	  cerr << "Stream reader " << myidx << " done spinning...\n";
	  io.unlock();
	}
      }
    }
    Image* image = winfo.images[framenumber%MAXFRAMES];
    int xres = image->get_xres();
    int yres = image->get_yres();
    while(idx < len){
      ImageHeader ih;
      if(idx+sizeof(ih) >= len){
	cerr << "protocol error, idx=" << idx << ", len=" << len << ", sizeof(ih)=" << sizeof(ih) << '\n';
	Thread::exitAll(1);
      }
      bcopy(&buf[idx], &ih, sizeof(ih));
      idx+=sizeof(ih);
      if(ih.row > yres || ih.col > xres || ih.channel == 3){
	cerr << "protocol error, ih=" << ih.row << ", " << ih.col << ", " << ih.channel << ", " << ih.numEncodings << '\n';
	cerr << "xres=" << xres << ", yres=" << yres << '\n';
	Thread::exitAll(1);
      }
      int n = ih.numEncodings;
      Pixel* imgrow = &(*image)(0, ih.row);
      int col = ih.col;
      int channel = ih.channel;
      for(int i=0;i<n;i++){
	unsigned char c = buf[idx++];
	if(c > 128){
	  // Run...
	  unsigned char value = buf[idx++];
	  int count = c-128;
	  if(col+count > xres){
	    cerr << "unpack error, col+" << col << ", count=" << count << ", xres=" << xres << '\n';
	    Thread::exitAll(1);
	  }
	  for(int i=0;i<count;i++)
	    imgrow[col+i][channel] = value;
	  col += count;
	} else {
	  // Data...
	  int count = c;
	  if(col+count > xres){
	    cerr << "unpack error, col+" << col << ", count=" << count << ", xres=" << xres << '\n';
	    Thread::exitAll(1);
	  }
	  for(int i=0;i<count;i++)
	    imgrow[col+i][channel] = buf[idx++];
	  col+= count;
	}
	if(idx > len){
	  cerr << "unpack error, idx=" << idx << ", len=" << len << '\n';
	  Thread::exitAll(1);
	}
      }
    }
  }
}


void setupStreams(WindowInfo& winfo, int newnstreams, RemoteReply& reply)
{
  int nstreams = (int)winfo.streams.size();
  for(int i=0;i<nstreams;i++)
    winfo.streams[i]->shutdown=true;
  winfo.streams.resize(newnstreams);
  winfo.streamthreads.resize(newnstreams);

  if(winfo.listen_sock != -1)
    close(winfo.listen_sock);

  // Create a single port
  winfo.listen_sock = socket(AF_INET, SOCK_STREAM, 0);

  struct sockaddr_in sa;
  sa.sin_family = AF_INET;
  int p;
  for(p = PORT; p<PORT+1024;p++){
    sa.sin_port = htons(p);
    sa.sin_addr.s_addr = 0;
    if(bind(winfo.listen_sock, (struct sockaddr*)&sa, sizeof(sa)) != -1)
      break;
  }

  cerr << "Listening for " << newnstreams << " streams on port " << p << '\n';
  if(listen(winfo.listen_sock, newnstreams) == -1){
    perror("listen");
    Thread::exitAll(1);
  }

  for(int i=0;i<newnstreams;i++)
    winfo.curFrames[i]=0;
  for(int i=0;i<newnstreams;i++){
    winfo.streams[i]=new RStream(winfo, winfo.listen_sock, i);
    winfo.streamthreads[i] = new Thread(winfo.streams[i],
					"Image receive thread");
    winfo.streamthreads[i]->detach();
    reply.ports[i] = p;
  }
  winfo.curFrame = 0;
  winfo.pendingCurFrame = 0;
}

void processMessage(WindowInfo& winfo, int sock)
{
  char buf[sizeof(RemoteMessage)];
  int total = 0;
  do {
    long n = read(sock, buf+total, sizeof(RemoteMessage)-total);
    if(n == 0){
      winfo.done=true;
      return;
    }
    if(n == -1){
      perror("3. read");
      Thread::exitAll(1);
    }
    total+=n;
  } while(total < (int)sizeof(RemoteMessage));
  RemoteMessage* rm = (RemoteMessage*)(&buf[0]);
  if(rm->len != sizeof(RemoteMessage)){
    cerr << "protocol error\n";
    Thread::exitAll(1);
  }
  cerr << "Got message, type=" << rm->type << '\n';
  RemoteReply reply;
  switch(rm->type){
  case RemoteMessage::CreateWindow:
    createWindow(winfo, rm->window);
    break;
  case RemoteMessage::ResizeWindow:
    resizeWindow(winfo, rm->xres, rm->yres);
    break;
  case RemoteMessage::SetupStreams:
    setupStreams(winfo, rm->nstreams, reply);
    break;
  };
  reply.len = sizeof(reply);
  long nwrite = write(sock, &reply, sizeof(reply));
  if(nwrite != sizeof(reply)){
    cerr << "write error";
    Thread::exitAll(1);
  }
  cerr << "Sent reply\n";
}

void serve(int inbound)
{
  WindowInfo winfo;
  // open window
  winfo.dpy=XOpenDisplay(NULL);
  if(!winfo.dpy){
    cerr << "Cannot open display\n";
    Thread::exitAll(1);
  }

  int error, event;
  if ( !glXQueryExtension( winfo.dpy, &error, &event) ) {
    cerr << "GL extension NOT available!\n";
    XCloseDisplay(winfo.dpy);
    winfo.dpy=0;
    Thread::exitAll(1);
  }
  winfo.screen=DefaultScreen(winfo.dpy);
  
  char* criteria="db, max rgb";
  if(!visPixelFormat(criteria)){
    cerr << "Error setting pixel format for visinfo\n";
    cerr << "Syntax error in criteria: " << criteria << '\n';
    Thread::exitAll(1);
  }
  int nvinfo;
  winfo.vi=visGetGLXVisualInfo(winfo.dpy, winfo.screen, &nvinfo);
  if(!winfo.vi || nvinfo == 0){
    cerr << "Error matching OpenGL Visual: " << criteria << '\n';
    Thread::exitAll(1);
  }
  winfo.cmap = XCreateColormap(winfo.dpy, RootWindow(winfo.dpy, winfo.screen),
			       winfo.vi->visual, AllocNone);

  int pipes[2];
  if(pipe(pipes) != 0){
    perror("pipe");
    Thread::exitAll(1);
  }
  winfo.sendPipe = pipes[1];
  winfo.recvPipe = pipes[0];

  fd_set fds;
  FD_ZERO(&fds);
  int* xfds;
  int numxfds;
  int maxfd=0;
  if(XInternalConnectionNumbers(winfo.dpy, &xfds, &numxfds) == 0){
    cerr << "XInternalConnectionNumbers failed\n";
    Thread::exitAll(1);
  }
  for(int i=0;i<numxfds;i++){
    if(xfds[i] > maxfd)
      maxfd=xfds[i];
    FD_SET(xfds[i], &fds);
  }
  if(inbound > maxfd)
    maxfd = inbound;
  FD_SET(inbound, &fds);
  if(winfo.recvPipe > maxfd)
    maxfd = winfo.recvPipe;
  FD_SET(winfo.recvPipe, &fds);

  XFlush(winfo.dpy);
  winfo.done=false;
  while(!winfo.done){
    fd_set readfds(fds);
    if(select(maxfd+1, &readfds, 0, 0, 0) == -1){
      if(errno == EINTR)
	continue;
      perror("select");
      Thread::exitAll(1);
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
    if(FD_ISSET(inbound, &readfds)){
      processMessage(winfo, inbound);
    }
    if(FD_ISSET(winfo.recvPipe, &readfds)){
      int b;
      long s = read(winfo.recvPipe, &b, sizeof(b));
      if(s != sizeof(b)){
	perror("4. read");
	Thread::exitAll(1);
      }
      while(winfo.curFrame < winfo.pendingCurFrame){
	Image* image = winfo.images[winfo.curFrame%MAXFRAMES];
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, image->get_xres(), 0, image->get_yres());
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.375, 0.375, 0.0);
	image->draw(0, false);
	glXSwapBuffers(winfo.dpy, winfo.win);
	glFinish();
	winfo.curFrame++;
      }
    }
  }
  if(winfo.listen_sock != -1)
    close(winfo.listen_sock);
  close(inbound);
  close(winfo.sendPipe);
  close(winfo.recvPipe);
  XCloseDisplay(winfo.dpy);
}
