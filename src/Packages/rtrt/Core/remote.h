
#ifndef remote_h
#define remote_h

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
};

struct ImageMessage {
  int len;
};

struct ImageHeader {
  unsigned int row:15;
  unsigned int col:15;
  unsigned int channel:2;
};

#endif
