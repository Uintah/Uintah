
#ifndef RSERVER_H
#define RSERVER_H

#include <Packages/rtrt/Core/remote.h>
#include <Core/Thread/WorkQueue.h>
#include <X11/Xlib.h>
#include <vector>

struct hostent;

namespace SCIRun {
  class Barrier;
}

namespace rtrt {
  class Image;
  class RemoteMessage;
  class RemoteReply;
  class Streamer;
  class RServer {
  public:
    RServer();
    ~RServer();

    void openWindow(Window win);
    void resize(int xres, int yres);
    void sendImage(Image* image, int nstreams);
  private:
    friend class Streamer;
    void send(RemoteMessage& msg);
    void send(RemoteMessage& msg, RemoteReply& reply);
    int control_sock;
    int nstreams;
    int ports[MAXSTREAMS];
    Image* image;
    struct ::hostent* hent;
    int frameno;

    SCIRun::Barrier* barrier;
    std::vector<Streamer*> streamers;
    SCIRun::WorkQueue work;
  };
}

#endif
