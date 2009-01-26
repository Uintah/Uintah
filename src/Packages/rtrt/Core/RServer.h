/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



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
