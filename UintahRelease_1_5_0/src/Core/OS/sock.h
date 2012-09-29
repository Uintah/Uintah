/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


//=======================
// sock.h
// David Hart
// Scientific Computing and Imaging,
// University of Utah,
//=======================

#ifndef Core_OS_sock_H
#define Core_OS_sock_H

#include <cstring>

#ifdef _WIN32
#include <winsock2.h>

#else

#define SOCKET int
#define SOCKET_ERROR -1
#define INVALID_SOCKET -1
#define SD_SEND 1

#include <unistd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <netdb.h>
#include <sys/ioctl.h>
#include <sys/errno.h>
#endif

namespace SCIRun {

//----------------------------------------------------------------------
// class socketinitializer
//----------------------------------------------------------------------
class SocketInitializer {
public:
  SocketInitializer();
  ~SocketInitializer();
};


//----------------------------------------------------------------------
// CLASS SOCKET
//----------------------------------------------------------------------
class Socket {
  
public:

  //-----------------
  // STATIC functions
  //-----------------

				// return the index of a socket with
				// data ready.  if block == true, this
				// will block till one of the sockets
				// has data, otherwise returns -1 if
				// no socket is ready
  static int
  FindReadyToRead(Socket** sockArray, int n, bool block);

  //------------------------
  // single socket functions
  //------------------------
  
  Socket();
  
  ~Socket();

				// 1 for synchronous-blocking,
				// 0 for asynchronous non-blocking
  void
  Block(int block);
  
				// connect to host
				// returns true on success, false
				// otherwise
  int
  ConnectTo(char* hostname, int port);

				// wait for connections from host
				// returns true on success, false
				// otherwise
  int
  ListenTo(char* hostname, int port);

				// start a new connection- returns
				// NULL if a connection couldn't be
				// made
  Socket*
  AcceptConnection();

				// returns true if this socket
				// currently has a valid connection
  int
  isConnected();

				// returns true is this socket is
				// currently in synchronous mode
  int
  isSynchronous();

				// returns true if there is data
				// waiting to be read on this socket
				// NON-blocking
  int
  isReadyToRead();

				// disconnect
  void			       
  Close();

				// disconnect and reopen a new socket
  void
  Reset();

  //-------------
  // IO functions
  //-------------

 				// read returns #bytes read if
				// successful, < 0 otherwise.
				// N.B.: 0 is a valid return value, not
				// an error - it means nothing was
				// read (in synchronous mode)
				// N.B.: these functions do the hton
				// and ntoh function calls for you.
  int Read(char*   location, int numBytes);
  int Read(int*    location, int numInts);
  int Read(float*  location, int numFloats);
  int Read(double* location, int numDoubles);
  
 				// read returns #bytes wrote if
				// successful, < 0 otherwise.
				// N.B.: 0 is a valid return value, not
				// an error - it means nothing was
				// read (in synchronous mode)
				// N.B.: these functions do the hton
				// and ntoh function calls for you.
  int Write(char*   location, int numBytes);
  int Write(int*    location, int numInts);
  int Write(float*  location, int numFloats);
  int Write(double* location, int numDoubles);

				// read and write null-terminated
				// strings.  Read allocates space for
				// the string.  You must deallocate it
				// yourself, if youre a good
				// programmer.  Read will NOT have
				// allocated a string for you if zero
				// or SOCKET_ERROR is returned
  int Read(char*& location);
  int Write(char* location);

				// read and write numbers
				// N.B.: these functions do the hton
				// and ntoh function calls for you.
  int Read(int& n);
  int Read(short& n);
  int Read(float& n);
  int Read(double& n);

  int Write(int n);
  int Write(short n);
  int Write(float n);
  int Write(double n);

  SOCKET fd;

protected:
  
  Socket(int fd);

				// true if socket is currently connected
  bool connected;

				// true if socket blocks on read when
				// no data is ready
  bool synchronous;

private:
  static SocketInitializer si;
  
};


//----------------------------------------------------------------------

} // End namespace SCIRun

#endif // Core_OS_sock_H
