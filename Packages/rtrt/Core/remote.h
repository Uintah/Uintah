
#ifndef remote_h
#define remote_h

#include <X11/Xlib.h>

namespace rtrt {

#define MAXSTREAMS 32
#define MAXBUFSIZE 65536
#define PORT 4300

  struct RemoteMessage {
    int len;
    enum MessageType {
      CreateWindow,
      ResizeWindow,
      SetupStreams
    };
    MessageType type;
    Window window;
    int xres, yres;
    int nstreams;
  };
  
  struct RemoteReply {
    int len;
    enum ReplyType {
    };
    int ports[MAXSTREAMS];
  };
  
  struct ImageMessage {
    int len;
  };
  
  struct ImageHeader {
    unsigned int row:15;
    unsigned int col:15;
    unsigned int channel:2;
    unsigned int numEncodings:16;
  };
}

// Protocol
// length, franeNumber
// (ImageHeader (encoding data)* num)*

#endif
