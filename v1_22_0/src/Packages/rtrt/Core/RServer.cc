
#include <Packages/rtrt/Core/RServer.h>
#include <Packages/rtrt/Core/remote.h>
#include <Packages/rtrt/Core/Image.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Mutex.h>
#include <sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <netdb.h>
#include <strings.h>
#include <sys/param.h>
#include <iostream>
#include <unistd.h>
using namespace std;
using namespace rtrt;
using namespace SCIRun;

extern Mutex io_lock_;

RServer::RServer()
  : work("remote image send")
{
  nstreams=0;
  barrier = new Barrier("RServer communication barrier");

  control_sock = socket(AF_INET, SOCK_STREAM, 0);
  if(control_sock == -1){
    perror("socket");
    Thread::exitAll(1);
  }
  char hostname[MAXHOSTNAMELEN];
  char* dpyname = getenv("DISPLAY");
  if(!dpyname){
    perror("rserver requires DISPLAY variable");
    Thread::exitAll(1);
  }
  char* p = dpyname;
  char* p2 = hostname;
  while(*p && *p != ':')
    *p2++ = *p++;
  *p2=0;
  if(strlen(hostname) == 0){
    cerr << "rserver doesn't work on local machine, dpy=" << dpyname << '\n';
    Thread::exitAll(1);
  }
  hent = gethostbyname(hostname);
  if(!hent){
    herror("gethostbyname");
    Thread::exitAll(1);
  }
  struct sockaddr_in sin;
  sin.sin_family = AF_INET;
  sin.sin_port = htons(PORT);
  bcopy(hent->h_addr_list[0], &sin.sin_addr.s_addr, hent->h_length);
  if(connect(control_sock, (struct sockaddr*)&sin, sizeof(sin)) == -1){
    perror("connect");
    Thread::exitAll(1);
  }
  cerr << "Successfully connected to rserver on host " << hostname << '\n';
}

void RServer::send(RemoteMessage& msg)
{
  RemoteReply reply;
  send(msg, reply);
}

void RServer::send(RemoteMessage& msg, RemoteReply& reply)
{
  cerr << "Sending message: type=" << msg.type << '\n';
  msg.len = sizeof(RemoteMessage);
  long n = write(control_sock, &msg, sizeof(msg));
  if(n != sizeof(RemoteMessage)){
    perror("write");
    Thread::exitAll(1);
  }
  cerr << "Waiting for reply\n";
  n = read(control_sock, &reply, sizeof(reply));
  if(n != sizeof(RemoteReply)){
    perror("read");
    Thread::exitAll(1);
  }
  if(reply.len != sizeof(RemoteReply)){
    cerr << "protocol error\n";
    Thread::exitAll(1);
  }
  cerr << "Got reply\n";
}

void RServer::openWindow(Window win)
{
  RemoteMessage msg;
  msg.type = RemoteMessage::CreateWindow;
  msg.window = win;
  send(msg);
}

void RServer::resize(int xres, int yres)
{
  RemoteMessage msg;
  msg.type = RemoteMessage::ResizeWindow;
  msg.xres = xres;
  msg.yres = yres;
  send(msg);
}

namespace rtrt {
  class Streamer : public Runnable {
  public:
    Streamer(int idx, RServer* rserver) 
      : idx(idx), rserver(rserver) {
      sock=-1;
      reconnect=true;
      bufsize=MAXBUFSIZE;
    }
    void sendImage();
    bool reconnect;
    int sock;
  private:
    unsigned char sendbuf[MAXBUFSIZE];
    int bufsize;
    void init();
    RServer* rserver;
    int idx;
    virtual void run();
  };
}

void RServer::sendImage(Image* image, int nstreams)
{
  double start = SCIRun::Time::currentSeconds();
  if(nstreams > MAXSTREAMS)
    nstreams=MAXSTREAMS;
  this->image = image;
  work.refill(image->get_yres(), nstreams);
  frameno++;
  if(nstreams != this->nstreams){
    cerr << "Adjusting streams to: " << nstreams  << '\n';
    RemoteMessage msg;
    msg.type = RemoteMessage::SetupStreams;
    msg.nstreams = nstreams;
    RemoteReply reply;
    send(msg, reply);
    for(int i=0;i<MAXSTREAMS;i++)
      ports[i] = reply.ports[i];

    for(int i=0;i<this->nstreams;i++){
      streamers[i]->reconnect=true;
      if(close(streamers[i]->sock) != 0){
	perror("close of streamer");
      }
    }
    frameno=0;
    int old_nstreams = this->nstreams;
    this->nstreams=nstreams;
    // Streams higher than nstreams will go away...
    if(old_nstreams != 0)
      barrier->wait(old_nstreams);
    streamers.resize(nstreams);
    for(int i=old_nstreams;i<nstreams;i++){
      streamers[i]=new Streamer(i, this);
      if(i != 0)
	new Thread(streamers[i], "Streamer thread");
    }
  } else {
    barrier->wait(nstreams);
  }
  streamers[0]->sendImage();
  barrier->wait(nstreams);
  this->image=0;
  double dt = SCIRun::Time::currentSeconds()-start;
  cerr << "Encoded/sent image in " << dt << " seconds\n";
}

void Streamer::sendImage()
{
  if(reconnect){
    if(sock == -1)
      close(sock);
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if(sock == -1){
      perror("socket");
      Thread::exitAll(1);
    }
    
    struct sockaddr_in sin;
    sin.sin_family = AF_INET;
    sin.sin_port = htons(rserver->ports[idx]);
    bcopy(rserver->hent->h_addr_list[0], &sin.sin_addr.s_addr,
	  rserver->hent->h_length);
    io_lock_.lock();
    cerr << "Streamer " << idx << " connecting to port " << rserver->ports[idx] << '\n';
    io_lock_.unlock();
    if(connect(sock, (struct sockaddr*)&sin, sizeof(sin)) == -1){
      perror("connect");
      Thread::exitAll(1);
    }
    io_lock_.lock();
    cerr << "Connect successful\n";
    io_lock_.unlock();
    reconnect=false;
  }

  Image* image = rserver->image;
  int xres = image->get_xres();
  int idx=0;
  int s, e;
  while(rserver->work.nextAssignment(s, e)){
    for(int row=s;row<e;row++){
      for(int channel=0;channel<3;channel++){
	// encode and send...
	if(idx+sizeof(ImageHeader)+16 >= bufsize){
	  // Send this packet...
	  int* len=(int*)(&sendbuf[0]);
	  *len = idx;
	  //io_lock_.lock();
	  //cerr << "1. sending " << idx << " bytes from streamer " << this->idx << "\n";
	  //io_lock_.unlock();
	  long n = write(sock, sendbuf, idx);
	  if(n != idx){
	    perror("write");
	    Thread::exitAll(1);
	  }
	  idx = 0;
	}
	if(idx == 0){
	  idx+=sizeof(int); // Reserve space for length
	  int* fn = (int*)(&sendbuf[idx]);
	  *fn = rserver->frameno;
	  idx+=sizeof(int);
	}
	ImageHeader ih;
	ih.row = row;
	ih.col = 0;
	ih.channel = channel;
	ih.numEncodings=0;
	int ihstart = idx;
	idx += (int)sizeof(ImageHeader);
	Pixel* imgrow = &(*image)(0, row);
	for(int col=0;col<xres;){
	  // Maybe flush the buffer first...
	  if(idx+2>=bufsize){
	    bcopy(&ih, &sendbuf[ihstart], sizeof(ImageHeader));
	    int* len = (int*)(&sendbuf[0]);
	    *len = idx;
	    //io_lock_.lock();
	    //cerr << "2. sending " << idx << " bytes from streamer " << this->idx << "\n";
	    //io_lock_.unlock();
	    long n = write(sock, sendbuf, idx);
	    if(n != idx){
	      perror("write");
	      Thread::exitAll(1);
	    }
	    
	    ih.row = row;
	    ih.col = col;
	    ih.channel = channel;
	    ih.numEncodings = 0;
	    idx=sizeof(int); // Reserve space for length
	    int* fn = (int*)(&sendbuf[idx]);
	    *fn = rserver->frameno;
	    idx+=sizeof(int);
	    ihstart = idx;
	    idx+=sizeof(ImageHeader);
	  }
	  if(col + 3 < xres && imgrow[col][channel] == imgrow[col+1][channel]
	     && imgrow[col][channel] == imgrow[col+2][channel]
	     && imgrow[col][channel] == imgrow[col+3][channel]){
            // A Run...
            int count=0;
	    unsigned char value = imgrow[col][channel];
	    while(col < xres && count < 127 && imgrow[col][channel] == value) {
	      count++; col++;
	    }
	    // A run of length count...
	    unsigned char op = 128+count;
	    sendbuf[idx++]=op;
	    sendbuf[idx++]=value;
	    ih.numEncodings++;
	  } else {
	    // Linear data...
	    int opidx = idx++;
            int count=0;
	    // A run of 4 is required before we should switch (3 is breakeven)
	    while(count < 127 && col < xres && idx < bufsize &&
		  (col+3 >= xres || 
		   !(imgrow[col][channel] == imgrow[col+1][channel] && 
		     imgrow[col][channel] == imgrow[col+2][channel] &&
		     imgrow[col][channel] == imgrow[col+3][channel]))) {
	      sendbuf[idx++]=imgrow[col++][channel];
	      count++;
	    }
	    sendbuf[opidx]=count;
	    ih.numEncodings++;
	  }
	}
	if(ih.numEncodings != 0){
	  bcopy(&ih, &sendbuf[ihstart], sizeof(ImageHeader));
	} else {
	  idx -= sizeof(ImageHeader);
	}
      }
    }
  }
  if(idx != 0){
    // Send this packet...
    int* len=(int*)(&sendbuf[0]);
    *len = idx;
    //io_lock_.lock();
    //cerr << "3. sending " << idx << " bytes from streamer " << this->idx << "\n";
    //io_lock_.unlock();
    long n = write(sock, sendbuf, idx);
    if(n != idx){
      perror("write");
      Thread::exitAll(1);
    }
  }
}

void Streamer::run()
{
  for(;;){
    // Send the image...
    sendImage();

    rserver->barrier->wait(rserver->nstreams);

    // Wait...
    rserver->barrier->wait(rserver->nstreams);
    if(idx >= rserver->nstreams){
      // We are no longer needed....
      return;
    }
  }
}
