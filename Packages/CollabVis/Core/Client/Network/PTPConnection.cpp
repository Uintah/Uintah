/*
 *
 * PTPConnection: Provides a point-to-point network connection
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: March 2001
 *
 */

#include <Network/PTPConnection.h>
#include <Logging/Log.h>
#include <Malloc/Allocator.h>

#include <sys/ioctl.h>
#include <sys/fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <signal.h>
#include <sys/types.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/poll.h>

namespace SemotusVisum {

using namespace std;

PTPConnection::PTPConnection() : theSocket( NO_SOCKET ) {

}

PTPConnection::PTPConnection( Socket theSocket ) : theSocket( theSocket ) {
  Log::log( ENTER, "[PTPConnection::PTPConnection] entered, thread id = " + mkString((int) pthread_self()) );
  // Now set the socket to blocking
  unsigned long temp = 1;
  if ( ioctl( theSocket, FIONBIO, &temp ) != 0 ) {
    Log::log( ERROR, string("[PTPConnection::PTPConnection] Couldn't set socket to blocking: ") +
	      strerror( errno ) );
  }
  Log::log( LEAVE, "[PTPConnection::PTPConnection] leaving, thread id = " + mkString((int) pthread_self()) );
}

PTPConnection::PTPConnection( string server, int port ) :
  theSocket( NO_SOCKET) {
  Log::log( ENTER, "[PTPConnection::PTPConnection] entered, thread id = " + mkString((int) pthread_self()) );
  // Connect to the given server:port
  unsigned int address;
  AddressValues( server.c_str(), &address, 1 );
  theSocket = ConnectToTcpPort( address, port );
  
  if ( theSocket == NO_SOCKET ) {
    Log::log( ERROR, string("[PTPConnection::PTPConnection] Couldn't open socket: ") +
	      strerror(errno) );
  }
  else {
    // Now set the socket to blocking
    unsigned long temp = 1;
    if ( ioctl( theSocket, FIONBIO, &temp ) != 0 ) {
      Log::log( ERROR, string("[PTPConnection::PTPConnection] Couldn't set socket to blocking: ") +
		strerror( errno ) );
    }
  }
  Log::log( LEAVE, "[PTPConnection::PTPConnection] leaving, thread id = " + mkString((int) pthread_self()) );
}

PTPConnection::~PTPConnection() {
  Log::log( ENTER, "[PTPConnection destructor] entered, thread id = " + mkString((int) pthread_self()) );
  // Close the socket.
  if ( theSocket != NO_SOCKET )
    Disconnect( &theSocket );
  Log::log( LEAVE, "[PTPConnection destructor] leaving, thread id = " + mkString((int) pthread_self()) );
}

  
int
PTPConnection::read ( char * data, int numBytes ) {
  Log::log( ENTER, "[PTPConnection::read] entered, thread id = " + mkString((int) pthread_self()) );

  cerr << "numBytes = " << numBytes << endl;

  /* Note - right now we do not switch from host-to-network byte order */
  Log::log( DEBUG, string("[PTPConnection::read] Reading ") + mkString(numBytes) +
	    "bytes on socket " + mkString(theSocket) );
  int bytesRead;

  struct pollfd fd;
  fd.fd = theSocket;
  fd.events = 0xff;
  
  int result = 0;

  result = poll( &fd, 1, 5 * 1000 );
  if ( result == 0 ) { // Timeout!
    std::cerr << "Timeout!" << endl;
    return -9;
  }
    

  bytesRead = ASIP_Receive( theSocket, (void *)data, numBytes );

  // Block until data is received

  while ( bytesRead < 0 && ( errno == EINTR || errno == EAGAIN ) ) {
    sleep(1);

    bytesRead = ASIP_Receive( theSocket, (void *)data, numBytes );
  }
  Log::log( LEAVE, "[PTPConnection::read] leaving, thread id = " + mkString((int) pthread_self()) );
  return bytesRead;
}

/* Convenience macro */
#ifndef timersub
#define	timersub(a, b, result)						      \
  do {									      \
    (result)->tv_sec = (a)->tv_sec - (b)->tv_sec;			      \
    (result)->tv_usec = (a)->tv_usec - (b)->tv_usec;			      \
    if ((result)->tv_usec < 0) {					      \
      --(result)->tv_sec;						      \
      (result)->tv_usec += 1000000;					      \
    }									      \
  } while (0)
#endif

int
PTPConnection::write( const char * data, int numBytes ) {
  Log::log( ENTER, "[PTPConnection::write] entered, thread id = " + mkString((int) pthread_self()) );
  /* Note - right now we do not switch from host-to-network byte order */
  
  int byteMax = 4096*8;
  int bytes;
  int start = numBytes;
  int result;
  struct timeval _start,_end,_result;

  gettimeofday( &_start,NULL );
  signal( SIGPIPE, SIG_IGN );
  
  //std::cerr << buffer << endl;
  Log::log( DEBUG, string("Writing ") + mkString(numBytes) +
	    " total on socket " + mkString(theSocket)   );
  
  while ( numBytes > 0 ) {
    
    bytes = ( numBytes > byteMax ) ? byteMax : numBytes;
    result = ASIP_Send( theSocket, (const void *)data, bytes );
    if ( result > 0 ) {
      numBytes -= result;
      data += result;
    }
    else {
      Log::log( DEBUG,  string("\t[PTPConnection::write] Send failed. Reason: ") + strerror( errno ));
      return -1;
    }
  }
  gettimeofday( &_end, NULL );
  timersub( &_end, &_start, &_result );
  Log::log( DEBUG, string("[PTPConnection::write] Done writing ") + mkString(start) +
	    " in " + mkString((float)_result.tv_sec*1000.0 +
			      (float)_result.tv_usec/1000.0) + " ms. ");
  Log::log( LEAVE, "[PTPConnection::write] leaving, thread id = " + mkString((int) pthread_self()) );
  return start;
}

bool
PTPConnection::isEqual( const Connection& c) {
  Log::log( ENTER, "[PTPConnection::isEqual] entered, thread id = " + mkString((int) pthread_self()) );
  // Equal if the underlying sockets are equal.
  return ( theSocket == ((PTPConnection *)(&c))->theSocket );
  Log::log( LEAVE, "[PTPConnection::isEqual] leaving, thread id = " + mkString((int) pthread_self()) );
}

void
PTPConnection::close() {
  Log::log( ENTER, "[PTPConnection::close] entered, thread id = " + mkString((int) pthread_self()) );
  if ( theSocket != NO_SOCKET )
    Disconnect( &theSocket );

  Log::log( LEAVE, "[PTPConnection::close] leaving, thread id = " + mkString((int) pthread_self()) );
}
 
}
