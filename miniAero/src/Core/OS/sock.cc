/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


//=======================
// sock.cpp
// David Hart
// Scientific Computing and Imaging
// University of Utah,
// department of Computer Science
//=======================


#include <iostream>
#include <signal.h>
#include <cstdio>

#ifdef _AIX
#  include <strings.h>  // for bzero
#endif

#include <Core/OS/sock.h>

#include <cerrno>

//----------------------------------------------------------------------

using namespace std;

namespace SCIRun {
  
// MAX -----------------------------------------------------------------
template <class T>
inline T MMAX(T a, T b) { 
  return ((a < b) ? b : a); 
}

//----------------------------------------------------------------------
SocketInitializer Socket::si;

//----------------------------------------------------------------------
SocketInitializer::SocketInitializer() {
  /* This ignores the SIGPIPE signal.  This is usually a good idea, since
     the default behavior is to terminate the application.  SIGPIPE is
     sent when you try to write to an unconnected socket.  You should
     check your return codes to make sure you catch this error! */

  struct sigaction sig;
  
  sig.sa_handler = SIG_IGN;
  sig.sa_flags = 0;
  sigemptyset(&sig.sa_mask);
  sigaction(SIGPIPE,&sig,NULL);

}

//----------------------------------------------------------------------
SocketInitializer::~SocketInitializer() {

    // shut down all socket resources for this application
}


//----------------------------------------------------------------------
int
Socket::FindReadyToRead(Socket** sockArray, int n, bool block) {

  fd_set readset;

  FD_ZERO(&readset);
  int i;
  SOCKET maxfd = 0;

				// clear the socket set
  for (i = 0; i < n; i++) {
    if (sockArray[i] && sockArray[i]->isConnected()) {
      FD_SET(sockArray[i]->fd, &readset);
      maxfd = MMAX(maxfd, sockArray[i]->fd);
    }
  }
  
				// N.B.: The first argument in select
				// might be different under UNIX

				// set timeval to zero, for
				// non-blocking
  timeval timeout;
  timeval* to = NULL;
  if (!block) {
    memset(&timeout, 0, sizeof(timeout));
    to = &timeout;
  }
  

				// if the last argument is NULL, make
				// this call Block until ready
  if (select(maxfd+1, &readset, NULL, NULL, to) == SOCKET_ERROR) {
    cerr << "Socket::FindReadyToRead: select() error" << endl;
  }

  
  for (i = 0; i < n; i++) {
    if (FD_ISSET(sockArray[i]->fd, &readset)) {
      return i;
    }
  }

  if (block) {
    cerr << "Socket: wierd: we should have had data here" << endl;
  }

				// no socket has data
  return -1;
}

//----------------------------------------------------------------------
				// see if this socket has data waiting
int Socket::isReadyToRead() {
  if (!connected) return false;
				// lets make it always non-blocking
				// for now...
  Socket* s = this;
  return (FindReadyToRead(&s, 1, 0) == 0);
}

//----------------------------------------------------------------------
Socket::Socket() {
  fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd == INVALID_SOCKET) {
    perror("Socket::Socket: Could not create socket");
  }

  // So that we can re-bind to it without TIME_WAIT problems
  
  // sources say to use this sparingly
  int reuse_addr = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse_addr,
    sizeof(reuse_addr));

  connected = false;
  synchronous = false;
}

//----------------------------------------------------------------------
Socket::Socket(int f) {
  fd = f;
  connected = false;
  synchronous = false;
}

//----------------------------------------------------------------------
Socket::~Socket() {
  Close();  
}

//----------------------------------------------------------------------
				// 1 for blocking, 0 for non-blocking
void
Socket::Block(int block) {
  //if (!connected) return;
  unsigned long bogus = (block) ? 0 : 1;
  synchronous = block;

  ioctl(fd, FIONBIO, &bogus);

				// these are apparently  valid,
				//  they call ioctl, i think
  //if (fcntl(s, F_SETFL, O_NDELAY) < 0) {
  //perror("fcntl F_SETFL, O_NDELAY");
  //}
  //if (fcntl(s, F_SETFL, O_NOBLOCK) < 0) {
  //perror("fcntl F_SETFL, O_NOBLOCK");
  //}
}

//----------------------------------------------------------------------
void
Socket::Close() {
  if (!connected) {
    return;
  }
  
  shutdown(fd, SD_SEND);
  close(fd);
  connected = false;
  
}

//----------------------------------------------------------------------
void
Socket::Reset() {

  Close();
  
  fd = socket(AF_INET, SOCK_STREAM, 0);

  if (fd == INVALID_SOCKET) {
    perror("Socket::Reset: Could not create socket");
    
  }
}

//----------------------------------------------------------------------
int
Socket::ConnectTo(char* hostname, int port) {
  int err;
  struct hostent* hostentry;
  hostentry = gethostbyname(hostname);
  struct sockaddr_in address;
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  address.sin_addr.s_addr = *((u_long*)(hostentry->h_addr_list[0]));
  for (int i = 0; i < 8; i++) {
    address.sin_zero[i] = 0;
  }
  
  err = connect(fd, (sockaddr *)(&address), sizeof(address));
  
  if (err == SOCKET_ERROR) {
    char errormsg[256];
    sprintf(errormsg,
      "Couldn't connect to %s:%d", hostname, port);
    perror(errormsg);
    Close();
    return false;
  }
  
  connected = true;
  return true;
  
}

//----------------------------------------------------------------------
int
Socket::ListenTo(char* hostname, int port) {
  struct hostent* hostentry;
  hostentry = gethostbyname(hostname);
  struct sockaddr_in address;
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  address.sin_addr.s_addr = *((u_long *)(hostentry->h_addr_list[0]));
  for (int i = 0; i < 8; i++) {
    address.sin_zero[i] = 0;
  }

  if (bind(fd,(sockaddr *)(&address),sizeof(address)) == SOCKET_ERROR) {
    char errormsg[256];
    sprintf(errormsg,
      "Socket::ListenTo: Could not listen to %s:%d", hostname, port);
    perror(errormsg);
    Close();
    return false;
  }
  
  listen(fd, 64);
  connected = true;
  return true;
}

//----------------------------------------------------------------------
Socket*
Socket::AcceptConnection() 
{
  
  SOCKET s = accept(fd, 0, 0);

  if (s == SOCKET_ERROR) {
    if (synchronous == false && errno == EWOULDBLOCK) {
      return NULL;
    }
    perror("Socket::AcceptConnection: Error accepting socket");
    return NULL;
  }

  Socket* sock = new Socket(s);
  sock->connected = true;
  return sock;
  
}

//----------------------------------------------------------------------
int
Socket::isConnected() {
  return connected;
}

//----------------------------------------------------------------------
int
Socket::isSynchronous() {
  return synchronous;
} 

//----------------------------------------------------------------------
				// these nasty union hacks allow me to
				// avoid nasty type-casting hacks
union floatintchar {
  float f;
  int i;
  char c;
};

union doubleintintchar {
  double d;
  int i[2];
  char c;
};



//----------------------------------------------------------------------
int
Socket::Read(char* location, int numBytes) {
  
  if (!connected) {
    return SOCKET_ERROR;
  }

  int r = 0;
  int amt;

  // recv may take multiple tries to get all the data
  while (r < numBytes) {
    // attempt to read
    amt = recv(fd, (location + r), numBytes - r, 0);

    // return if there was an error
    if (amt == SOCKET_ERROR || amt == 0) {
      // if there was an error, then we are probably no longer connected, unless
      // blocking is turned off
      if (synchronous == false && errno == EWOULDBLOCK) {
        return 0;
      } else {
        Close();
        return SOCKET_ERROR;
      }
    }

    r += amt;
  }
  return r;
  
}

//----------------------------------------------------------------------
int
Socket::Write(char* location, int numBytes) {

  if (!connected) {
    return SOCKET_ERROR;
  }

  int s = 0;
  int amt;
  // send may take multiple tries
  while (s < numBytes) {
    // attempt to send
    amt = send(fd, (location + s), numBytes - s, 0);
    // return if there was an error
    if (amt == SOCKET_ERROR || amt == 0) {
      // if there was an error, then we are no longer connected
      if (synchronous == false && errno == EWOULDBLOCK) {
        return 0;
      } else {
        Close();
        return SOCKET_ERROR;
      }
    }
    s += amt;
  }
  return s;
  
}

//----------------------------------------------------------------------
int
Socket::Read(int* location, int numInts) {
  int i;
  floatintchar* l = (floatintchar*)location;
				// send
  int retval = Read((char*)location, numInts*sizeof(int));
				// convert back to host order
  for (i = 0; i < numInts; i++) {
    l[i].i = htonl(l[i].i);
  }
  return retval;
}

int
Socket::Read(float* location, int numFloats) {
  int i;
  floatintchar* l = (floatintchar*)location;
				// send
  int retval = Read((char*)location, numFloats*sizeof(float));
				// convert back to host order
  for (i = 0; i < numFloats; i++) {
    l[i].i = htonl(l[i].i);
  }
  return retval;
}

int
Socket::Read(double* location, int numDoubles) {
  int i, temp;
  doubleintintchar* l = (doubleintintchar*)location;
  
				// send
  int retval = Read((char*)location, numDoubles*sizeof(double));
  
				// convert back to host order
  for (i = 0; i < numDoubles; i++) {
    l[i].i[0] = htonl(l[i].i[0]);
    l[i].i[1] = htonl(l[i].i[1]);
    temp = l[i].i[0]; l[i].i[0] = l[i].i[1]; l[i].i[1] = temp;
  }
  return retval;
}

//----------------------------------------------------------------------
int
Socket::Write(int* location, int numInts) {
  int i;
  floatintchar* l = (floatintchar*)location;
				// convert to network order
  for (i = 0; i < numInts; i++) {
    l[i].i = htonl(l[i].i);
  }
				// send
  int retval = Write((char*)location, numInts*sizeof(int));
				// convert back to host order
  for (i = 0; i < numInts; i++) {
    l[i].i = htonl(l[i].i);
  }
  return retval;
}

int
Socket::Write(float* location, int numFloats) {
  int i;
  floatintchar* l = (floatintchar*)location;
				// convert to network order
  for (i = 0; i < numFloats; i++) {
    l[i].i = htonl(l[i].i);
  }
				// send
  int retval = Write((char*)location, numFloats*sizeof(float));
				// convert back to host order
  for (i = 0; i < numFloats; i++) {
    l[i].i = htonl(l[i].i);
  }
  return retval;
}

int
Socket::Write(double* location, int numDoubles) {
  int i, temp;
  doubleintintchar* l = (doubleintintchar*)location;
				// convert to network order
  for (i = 0; i < numDoubles; i++) {
    l[i].i[0] = htonl(l[i].i[0]);
    l[i].i[1] = htonl(l[i].i[1]);
    temp = l[i].i[0]; l[i].i[0] = l[i].i[1]; l[i].i[1] = temp;
  }
				// send
  int retval = Write((char*)location, numDoubles*sizeof(double));
				// convert back to host order
  for (i = 0; i < numDoubles; i++) {
    l[i].i[0] = htonl(l[i].i[0]);
    l[i].i[1] = htonl(l[i].i[1]);
    temp = l[i].i[0]; l[i].i[0] = l[i].i[1]; l[i].i[1] = temp;
  }
  return retval;
}

//----------------------------------------------------------------------
int
Socket::Read(char*& location) {
  int len = 0;
  int r1=0, r2=0;
  int rtemp=0;

				// read the length of the string
  r1 = Read(len);
  
  if (r1 == 0) return 0;
  else if (r1 != sizeof(len)) return SOCKET_ERROR;
  
				// allocate space for string plus null
				// char
  location = new char[len+1];

				// read a string unless an error
				// occurs.  for string read in
				// asynchronous mode, we'll ignore a
				// read return value of zero since the
				// whole string should get here sooner
				// or later

  while (r2 < len) {
				// read the string itself
    rtemp = Read(location+r2, len-r2);
    if (rtemp == SOCKET_ERROR) return SOCKET_ERROR;
    r2 += rtemp;
    
  }
  
				// null-terminate
  location[len] = 0;
  
  return r1+r2;
}

//----------------------------------------------------------------------
int
Socket::Write(char* location) {
  
				// get and write string length
  int len = strlen(location);
  int r1, r2;
  r1 = Write(len);
  if (r1 == SOCKET_ERROR) return SOCKET_ERROR;
  
				// write the string
  r2 = Write(location, len);
  if (r2 == SOCKET_ERROR) return SOCKET_ERROR;
  return r1+r2;
  
}

//----------------------------------------------------------------------
int
Socket::Read(int& n) {
  int retval = Read((char*)&n, sizeof(n));
  n = ntohl(n);
  return retval;
}

int
Socket::Read(short& n) {
  int retval = Read((char*)&n, sizeof(n));
  n = ntohs(n);
  return retval;
}

int
Socket::Read(float& n) {
  floatintchar fin;
  int retval = Read(&fin.c, sizeof(float));
  fin.i = ntohl(fin.i);
  n = fin.f;
  return retval;
}

int
Socket::Read(double& n) {
  doubleintintchar diin;
  int retval = Read(&diin.c, sizeof(double));
  int temp = diin.i[0]; diin.i[0] = diin.i[1]; diin.i[1] = temp;
  diin.i[0] = ntohl(diin.i[0]);
  diin.i[1] = ntohl(diin.i[1]);
  n = diin.d;
  return retval;
}

//----------------------------------------------------------------------
int
Socket::Write(int n) {
  n = htonl(n);
  return Write((char*)&n, sizeof(n));
}

int
Socket::Write(short n) {
  n = htons(n);
  return Write((char*)&n, sizeof(n));
}

int
Socket::Write(float n) {
  floatintchar fin;
  fin.f = n;
  fin.i = htonl(fin.i);
  return Write(&fin.c, sizeof(float));
}

int
Socket::Write(double n) {
  doubleintintchar diin;
  diin.d = n;
  diin.i[0] = htonl(diin.i[0]);
  diin.i[1] = htonl(diin.i[1]);
  int temp = diin.i[0]; diin.i[0] = diin.i[1]; diin.i[1] = temp;
  return Write(&diin.c, sizeof(double));
}


} // End namespace SCIRun
