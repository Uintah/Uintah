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
//  
//    File   : Socket.h
//    Author : Martin Cole
//    Date   : Fri Aug 18 10:49:44 2006

//! a good source for socket communication.
//! http://en.wikipedia.org/wiki/Berkeley_sockets


#if !defined(Socket_h)
#define Socket_h

#ifdef _WIN32
#  include <winsock2.h>
#else
#  include <sys/types.h>
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <netdb.h>
#  include <unistd.h>
#  include <arpa/inet.h>
#endif
#include <string>

namespace SCIRun {

class Socket
{
 public:
  Socket();
  virtual ~Socket();

  //! Set the descriptor, and set socket options.
  bool create();
  //! Bind the descriptor to an address.
  bool bind(const int port);
  //! Set the socket to listen mode, ready to accept connections.
  bool listen() const;
  //! After listening, accept a connection and return in conn
  //! the new accepted connection socket.
  bool accept(Socket &conn) const;

  // Connect to a listening socket. (typical for client)
  // host can be a named host or ip.
  bool connect(const std::string host, const int port);

  //! Send string message.
  bool write(const std::string) const;
  //! Send specified amount of data from buf.
  bool write(const void *buf, size_t bytes) const;

  //! Read string message from the socket. Use this for control messages.
  //! Read up to the first '\n' character, and stuff into the input s.
  //! ('Read' DOES NOT RETURN UNTIL A '\n' IS READ.)
  int read(std::string &s) const;

  //! Read specified amount of data. buf must be allocated by caller, 
  //! and must be big enough to hold the read data. bytes specifies how 
  //! much expected data to read, and the socket will block until that much 
  //! data is recieved. Use this for large data passing.
  int read(void* buf, size_t bytes) const;

  //! Set the socket blocking status to correspond with the passed in bool.
  //! default is blocking.
  void set_blocking(const bool);

  //! Test for valid socket descriptor.
  bool is_valid() const { return sock_ != -1; }

  //! Returns a string of the form "HostName (port #)"
  std::string getSocketInfo();

 private:
  char            *buf_;
  //! The socket descriptor.
  int              sock_;
  //! Struct that holds the address.
  sockaddr_in      addr_;
};

} // namespace SCIRun

#endif //Socket_h
