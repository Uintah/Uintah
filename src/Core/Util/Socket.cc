/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

//    File   : Socket.cc
//    Author : Martin Cole
//    Date   : Fri Aug 18 10:53:33 2006


#include <Core/Util/Socket.h>

#include <sstream>
#include <string>

#include <cstring>
#include <cerrno>
#include <cstdio>

#ifdef _WIN32
#  define socklen_t int
#  define close closesocket
#else
#  include <fcntl.h>
#endif

#if defined(__APPLE__)
#  define SOCKET_NOSIGNAL SO_NOSIGPIPE
#elif defined(_WIN32) || defined(__sgi) || defined(_AIX)
#  define SOCKET_NOSIGNAL 0
#else
#  define SOCKET_NOSIGNAL MSG_NOSIGNAL
#endif

const int MAXCONNECTIONS = 25;
const int MAXRECV = 1024;

namespace SCIRun {

using namespace std;

Socket::Socket() :
  buf_(new char[MAXRECV]),
  sock_(-1)
{
  memset(&addr_, 0, sizeof(addr_));
}

Socket::~Socket()
{
  if (is_valid()) {
    ::close(sock_);
    sock_ = -1;
  }
}

bool 
Socket::create()
{
  sock_ = socket(AF_INET, SOCK_STREAM, 0);
  if (! is_valid()) { return false; }

  int on = 1;
  if (setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, 
		 (const char*) &on, sizeof(on)) == -1)
  {
    perror("ERROR in setsockopt()");
    return false;
  }

  return true;
}

bool 
Socket::bind (const int port)
{
  if (! is_valid())  { return false; }

  addr_.sin_family = AF_INET;
  addr_.sin_addr.s_addr = INADDR_ANY;
  //! make sure port is in network byte order, use htons.
  addr_.sin_port = htons(port);

  // if bind fails it returns -1
  if (::bind(sock_, (struct sockaddr*) &addr_, sizeof(addr_)) == -1)
  {
    perror("ERROR in bind()");
    return false;
  }
  return true;
}


bool 
Socket::listen() const
{
  if (! is_valid()) { return false; }

  if (::listen(sock_, MAXCONNECTIONS) == -1) {
    perror("ERROR in listen()");
    return false;
  }

  return true;
}

bool 
Socket::accept(Socket& new_socket) const
{
  int addr_length = sizeof(new_socket.addr_);
  new_socket.sock_ = ::accept(sock_, (sockaddr*)&new_socket.addr_, 
			      (socklen_t*) &addr_length);
  
  if (new_socket.sock_ <= 0) { 
    perror("ERROR in accept()");
    return false; 
  }
  return true;
}

bool 
Socket::write(const std::string s) const
{
  int sent = ::send(sock_, s.c_str(), s.size(), SOCKET_NOSIGNAL);

  if (sent == -1) { 
    perror("ERROR in write(const std::string s)");
    return false; 
  }
  return true;
}

bool 
Socket::write(const void *buf, size_t bytes) const
{
#ifdef _WIN32
  // windows wants a char buffer
  const char* charbuf = (const char*) buf;
  bytes *= (sizeof(void*)/sizeof(char*));
  int sent = ::send(sock_, charbuf, bytes, SOCKET_NOSIGNAL);
#else
  int sent = ::send(sock_, buf, bytes, SOCKET_NOSIGNAL);
#endif

  if (sent == -1) { 
    perror("ERROR in write(const void *buf, size_t bytes)");
    return false; 
  }
  return true;
}

int 
Socket::read(std::string& s) const
{
  char c = 0;
  s = "";
  int total = 0;
  int read = 1;
  //cerr << "starting to recieve characters" << endl;
  while (read == 1 && c != '\n') {
    read = ::recv(sock_, &c, 1, 0);
    //cerr << "got a: '" << c << "'" << endl;
    if (read == -1) {
      perror("ERROR in read(std::string& s)");
      return 0;
    } else if (read == 0)  {
      return 0;
    } else  {
      s = s + c;
      total++;
    }
  }
  return total;
}

int 
Socket::read(void* buf, size_t bytes) const
{
  //cerr << "starting to recieve " << bytes << endl;
  size_t total_read = 0;
  char* cur = (char*)buf;
  while (total_read < bytes) {

    int read = ::recv(sock_, cur, bytes, 0);
    
    if (read == -1) {
      perror("ERROR in read(void* buf, size_t bytes)");
      return 0;
    } else if (read == 0)  {
      return 0;
    } else  {
      //cerr << "got " << read << " bytes of data" << endl;
      total_read += read;
      cur += read;
    }
  }
  //cerr << "recieved " << total_read << endl;
  return total_read;
}

bool 
Socket::connect(const std::string host, const int port)
{
  if (! is_valid()) return false;

  addr_.sin_family = AF_INET;
  addr_.sin_port = htons(port);

  struct hostent* hostentry;
  hostentry = gethostbyname(host.c_str());
  if (hostentry == 0) {
    printf( "ERROR in connect(): %d.  Check in netdb.h for what this number means.\n", h_errno );
    return false;
  }
  addr_.sin_addr.s_addr = *((u_long*)(hostentry->h_addr_list[0]));

  int status = ::connect(sock_, (sockaddr*)&addr_, sizeof(addr_));

  if (status == 0) { return true; }

  perror("ERROR 2 in connect()");
  return false;
}

void 
Socket::set_blocking(const bool b)
{
#ifndef _WIN32
  int opts;
  opts = fcntl(sock_, F_GETFL);
  
  if (opts < 0) { 
    perror("ERROR in set_non_blocking() get");
    return; 
  }
  
  if (! b) {
    opts = ( opts | O_NONBLOCK );
  } else {
    opts = ( opts & ~O_NONBLOCK );
  }
  
  if (fcntl(sock_, F_SETFL, opts) < 0) {
    perror("ERROR in set_non_blocking() set");
  }
#else
  unsigned long val = !b;
  if (ioctlsocket(sock_, FIONBIO, &val) < 0) {
    perror("ERROR in set_non_blocking() set");
  }
#endif
}

string
Socket::getSocketInfo() 
{
  ostringstream info;
  info << inet_ntoa( addr_.sin_addr ) << "(" << ntohs( addr_.sin_port ) << ")";
  return info.str();
}

} //namespace SCIRun
