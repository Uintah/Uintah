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
namespace Network {

using namespace SemotusVisum::Logging;
using namespace std;

//////////
// Instantiation of connection list.
list<PTPConnection*>
PTPConnection::connectionList;

//////////
// Instantiation of the lock for the connection list.
CrowdMonitor
PTPConnection::connectionListLock( "ConnectionListLock" );

//////////
// Instantiation of the flag for 'connection list changed'.
bool
PTPConnection::listChanged = true;

//////////
// Instantiation of the list of sockets available from the current
// connections.
Socket *
PTPConnection::socketList = NULL;

PTPConnection::PTPConnection() : theSocket( NO_SOCKET ) {

}

PTPConnection::PTPConnection( Socket theSocket ) : theSocket( theSocket ) {

  cerr << "In PTPConnection::PTPConnection, thread id is " << pthread_self() << endl;  
  // Add ourselves to the list
  PTPConnection::connectionListLock.writeLock();
  listChanged = true;
  connectionList.push_front( this );
  PTPConnection::connectionListLock.writeUnlock();

  // Now set the socket to blocking
  unsigned long temp = 1;
  if ( ioctl( theSocket, FIONBIO, &temp ) != 0 ) {
    char buf[100];
    snprintf( buf, 100, "Couldn't set socket to blocking: %s",
	     strerror( errno ) );
    Log::log( Logging::ERROR, buf );
  }

  // Disable Nagle's Algorithm
#if 0
  int delay = true;
  if ( setsockopt( theSocket,
		   SOL_TCP,
		   TCP_NODELAY,
		   &delay,
		   sizeof( int ) ) < 0 ) {
    char buffer[1000];
    snprintf( buffer, 1000, "Couldn't set TCP to no delay: %s",
	      strerror( errno ) );
    std::cerr << buffer << endl;
    //    Log::Log( Logging::WARNING, buffer );
  }

#endif
  cerr << "End of PTPConnection::PTPConnection, thread id is " << pthread_self() << endl;  

}

PTPConnection::~PTPConnection() {
  cerr << "In PTPConnection destructor, thread id is " << pthread_self() << endl;  
  // Close the socket.
  if ( theSocket != NO_SOCKET )
    Disconnect( &theSocket );

  // Remove ourselves from the list.
  PTPConnection::connectionListLock.writeLock();
  listChanged = true;
  connectionList.remove( this );
  PTPConnection::connectionListLock.writeUnlock();
  cerr << "End of PTPConnection destructor, thread id is " << pthread_self() << endl;  
}

  
int
PTPConnection::read ( char * data, int numBytes ) {
  cerr << "In PTPConnection::read, thread id is " << pthread_self() << endl;  
  /* Note - right now we do not switch from host-to-network byte order */
  char buffer[1000];
  snprintf( buffer, 1000, "Reading %d bytes on socket %d",
	    numBytes, theSocket );
  Log::log( Logging::DEBUG, buffer );
  int bytesRead;
  int calls = 1;
#if 0
  int result = 0;
  //while ( result == 0 ) {
  result = CheckIfReadable( theSocket, 100 );
  std::cerr << "Read result for socket " << theSocket << " : " <<
    result << endl;
  //}
#else
  struct pollfd fd;
  fd.fd = theSocket;
  fd.events = 0xff;
  
  int result = 0;

  result = poll( &fd, 1, 5 * 1000 );
  if ( result == 0 ) { // Timeout!
    std::cerr << "Timeout!" << endl;
    return -9;
  }
    
  //  std::cerr << "Poll result: " << result << endl;
  //  if ( fd.revents & POLLIN )
  //   std::cerr << "\tThere is data to be read!" << endl;
#endif
#if 1
  bytesRead = ASIP_Receive( theSocket, (void *)data, numBytes );

  // Block until data is received

  while ( bytesRead < 0 && ( errno == EINTR || errno == EAGAIN ) ) {
    sleep(1);

    bytesRead = ASIP_Receive( theSocket, (void *)data, numBytes );
    calls++;
  }
#endif
  cerr << "End of PTPConnection::read, thread id is " << pthread_self() << endl;  
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
  cerr << "In PTPConnection::write, thread id is " << pthread_self() << endl;  
  /* Note - right now we do not switch from host-to-network byte order */
  
  cerr << "Sending " << numBytes << " bytes" << endl;

  //cerr << "Data: " << data << endl;

#if 1
  int byteMax = 4096*8;

  cerr << "Maximum message size: " << byteMax << endl;
  int bytes;
  int start = numBytes;
  int result;
  struct timeval _start,_end,_result;

  gettimeofday( &_start,NULL );
  char buffer[1000];
  signal( SIGPIPE, SIG_IGN );
  snprintf(buffer, 1000,
	   "Writing %d total on socket %d", numBytes, theSocket);
  
  //std::cerr << buffer << endl;
  Log::log( Logging::DEBUG, buffer );
  
  while ( numBytes > 0 ) {
    
    bytes = ( numBytes > byteMax ) ? byteMax : numBytes;
    result = ASIP_Send( theSocket, (const void *)data, bytes );
    cerr << "Called ASIP_Send for " << bytes << " bytes of data" << endl;    

    if ( result > 0 ) {
      numBytes -= result;
      data += result;
    }
    else {
      snprintf( buffer, 1000, "\tSend failed. Reason: %s", strerror( errno ) );
      Log::log( Logging::DEBUG, buffer );
      std::cerr << buffer << endl;
      std::cerr << "\tWe've written " << start-numBytes << " bytes" << endl;
      return -1;
    }
#if 0
    if ( result < 0 )
      std::cerr << "\tSend failed. Reason: " << strerror( errno ) << endl;
    else 
      std::cerr << "\tWrote " << result << " bytes. Have " <<
	numBytes << " bytes left. " << endl;
#endif
    ///    else 
    //  if ( calls-- == 0 )
    //	return result;
  }
  gettimeofday( &_end, NULL );
  timersub( &_end, &_start, &_result );
  snprintf( buffer, 1000, "Done writing %d bytes in %0.3f ms. ", start,
	    (float)_result.tv_sec*1000.0 + (float)_result.tv_usec/1000.0 );
 
  Log::log( Logging::DEBUG,  buffer );

  cerr << "End of PTPConnection::write, thread id is " << pthread_self() << endl;  


  return start;
#else
  
  int sendResult = ASIP_Send( theSocket, (const void *)data, numBytes );

  cerr << "End of PTPConnection::write, thread id is " << pthread_self() << endl;  

  return sendResult;
#endif
}


Connection **
PTPConnection::getReadyToRead() {
  cerr << "In PTPConnection::getReadyToRead, thread id is " << pthread_self() << endl;  
  cerr << "End of PTPConnection::getReadyToRead, thread id is " << pthread_self() << endl;  
  return PTPConnection::getReadyConns();
  
}

bool
PTPConnection::isEqual( const Connection& c) {
  // Equal if the underlying sockets are equal.
  cerr << "In PTPConnection::isEqual, thread id is " << pthread_self() << endl;  
  cerr << "End of PTPConnection::isEqual, thread id is " << pthread_self() << endl;  
  return ( theSocket == ((PTPConnection *)(&c))->theSocket );

}

void
PTPConnection::close() {
  cerr << "In PTPConnection::close, thread id is " << pthread_self() << endl;  
  if ( theSocket != NO_SOCKET )
    Disconnect( &theSocket );
  cerr << "End of PTPConnection::close, thread id is " << pthread_self() << endl;  
}
  
  
Connection **
PTPConnection::getReadyConns() {
  cerr << "In PTPConnection::getReadyConns, thread id is " << pthread_self() << endl;  
  /* If the connection list has changed since we were last called,
     rebuild the array of sockets. */
  PTPConnection::connectionListLock.readLock();
  if ( listChanged ) {

    Log::log( Logging::DEBUG, "List of ready-to-read sockets changed.");
    
    // Clean up the old list if it existed.
    if (socketList) delete socketList;

    // Resize and repopulate the list.
    socketList = scinew Socket[ connectionList.size() + 1 ];
    list<PTPConnection*>::iterator li;
    int i;
    for (i = 0, li = connectionList.begin();
	 i < connectionList.size();
	 i++, li++)
      socketList[ i ] = (*li)->theSocket;
    socketList[ connectionList.size() ] = NO_SOCKET;
    
    PTPConnection::connectionListLock.readUnlock();

    // Now set our 'list changed' flag to false.
    PTPConnection::connectionListLock.writeLock();
    listChanged = false;
    PTPConnection::connectionListLock.writeUnlock();
    
    /* Note - It is true that we might get another connection between now
       and when we read. This is not important to us, at least for right
       now. 
    for ( int j = 0; j < i; j++ )
      cerr << "SL[" << j << "] = " << socketList[ j ] << endl;
    */
  }
  else 
    PTPConnection::connectionListLock.readUnlock();


  /* Now check for any readable sockets. First, we pull the list of
     available sockets. If we have none, we return immediately. If we do
     have sockets with data, we build a connection list of the appropriate
     size, and match sockets with connections.
  */
  Socket * sockets = scinew Socket[ connectionList.size() ];
  int numReadable = CheckIfAnyReadable( socketList,
					sockets,
					Timeout );

  // No sockets with data available.
  if ( numReadable == 0 ) {
    delete sockets;
    return NULL;
  }
  // Error!
  if (numReadable < 0 ) {
    delete sockets;
    char buffer[ 256 ];
    sprintf( buffer, "Error in CheckIfAnyReadable: %s",
	     strerror( errno ) );
    Log::log( Logging::ERROR, buffer );
    return NULL;
  }

  PTPConnection ** connections = scinew PTPConnection*[ numReadable + 1 ];

  // As numReadable <= # connections, we go by numReadable...
  PTPConnection::connectionListLock.readLock();
  
  list<PTPConnection *>::iterator li;
  for (int i = 0; i < numReadable; i++ ) {
    for ( li = connectionList.begin(); li != connectionList.end(); li++ )
      if ( sockets[ i ] == (*li)->theSocket ) {
	connections[ i ] = (*li);
	break;
      }
  }
  ConvertData( NULL, NULL, NULL, 0, 0 );

  delete sockets;
  connections[ numReadable ] = NULL;
  
  PTPConnection::connectionListLock.readUnlock();
  cerr << "End of PTPConnection::getReadyConns, thread id is " << pthread_self() << endl;  
  return (Connection **)connections;

  
  
}


}
}

