/*
 * SERVER
 * 
 * MulticastConnection: Provides a multicast network connection
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: March 2001
 *
 */

#include <sys/socket.h>  /* AF_INET */
#include <netinet/in.h>  /* IPPROTO_TCP struct in_addr */
#include <string.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <iostream.h>
#include <Network/MulticastConnection.h>
#include <Network/Connection.h>
#include <Logging/Log.h>

namespace SemotusVisum {
namespace Network {

using namespace Logging;
using namespace std;


MulticastConnection::MulticastConnection() : sendfd(-1) {

}


MulticastConnection::MulticastConnection( const char * group,
					  const int port ) {

  cerr << "In MulticastConnection::MulticastConnection, thread id is " << pthread_self() << endl;  

  /*
  if(SV_TRANSFER_MODE == 1){
    //
    // Reliable Multicasting Code
    //

    // This has been removed

  }
  else if(SV_TRANSFER_MODE == 2){

    //
    // Unreliable, Low-Level Multicasting Code - broken
    //

    // Initialize the connection 

    // Nothing special is required to send a multicast datagram.
    // The only things that need to be done are setting the following
    // socket options:
    // - outgoing interface
    // - ttl
    // - loopback

    sendfd = Udp_client("228.6.6.6", "6210", (void **) &sasend, &salen);

    // turn loopback on
    Mcast_set_loop(sendfd, 1);

    // set ttl
    Mcast_set_ttl(sendfd, 5);


    // Initialize the connection 
  
    // create what looks like an ordinary UDP socket 
    if ((sock=socket(AF_INET,SOCK_DGRAM,0)) < 0) {
      Log::log( ERROR,
	      "Error creating multicast socket" );
      Log::log( ERROR, strerror( errno ) );
      return;
    }
 
    // set up destination address 
    memset(&addr,0,sizeof(addr));
    addr.sin_family=AF_INET;
    addr.sin_addr.s_addr=inet_addr(group);
    addr.sin_port=htons(port);
  
    // Set the ttl to 5
    int ttl = 5;
    if ( setsockopt(sock, IPPROTO_IP, IP_TTL,
		  (void *) &ttl, sizeof(ttl) ) == -1 ) {
      Log::log( ERROR,
	      "Error setting ttl on multicast socket" );
      Log::log( ERROR, strerror( errno ) );
      cerr << "TTL " << strerror( errno ) << endl;  
      return;
    }
    // We're ready to go... 
   
  }
  else{
    Log::log( ERROR,
	      "Unrecognized multicast transfer mode" );
  }
  */
  cerr << "End of MulticastConnection::MulticastConnection, thread id is " << pthread_self() << endl;
  
}

  
MulticastConnection::~MulticastConnection() {
  cerr << "In MulticastConnection destructor, thread id is " << pthread_self() << endl;  
  /*
  if(SV_TRANSFER_MODE == 1){
    //
    // Reliable Multicasting Code
    //

    // This has been removed

  }
  else if(SV_TRANSFER_MODE == 2){
    //
    // Unreliable, Low-Level Multicasting Code - broken
    //
  
    
    //if ( sock != -1 )
    //  ::close( sock );
    

    if(sendfd != -1){
      Close(sendfd);
    }
  }
  else{
    Log::log( ERROR,
	      "Unrecognized multicast transfer mode" );
  }
  */
  cerr << "End of MulticastConnection destructor, thread id is " << pthread_self() << endl;  
}

  
int
MulticastConnection::read( char * data, int numBytes ) {
  cerr << "In MulticastConnection::read, thread id is " << pthread_self() << endl;  
  cerr << "End of MulticastConnection::read, thread id is " << pthread_self() << endl;  
  /* We do no reading.... */
  return 0;
  
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
MulticastConnection::write( const char * data, int numBytes ) {
  cerr << "In MulticastConnection::write, thread id is " << pthread_self() << endl;  
  /*
  if(SV_TRANSFER_MODE == 1){
    //
    // Reliable Multicasting Code
    //

    // This has been removed

  }
  else if(SV_TRANSFER_MODE == 2){

    //
    // Unreliable, Low-Level Multicasting Code
    //
  
    unsigned int numBytesLeft, bytes;
    //static char		line[MAXLINE];		// hostname and process ID 
    //char data[numBytes];
 
    const char * ptr;	
    //struct utsname	myname;
    //struct tms dummy;
    //double start, end;
    //double ci = 1./CLK_TCK;

    printf("In send_all\n");

    //if (uname(&myname) < 0){
    //  err_sys("uname error");
    //}
    //snprintf(data, sizeof(data), "%s, %d\n", myname.nodename, getpid());


    //start = (double)times(&dummy)*ci;
    
    //
    // Send data 
    //

    printf("Sending %d total bytes of data\n", numBytes);

    numBytesLeft = numBytes;
    ptr = data;

    while ( numBytesLeft > 0 ) {
  
      bytes = ( numBytesLeft > MAXLINE ) ? MAXLINE : numBytesLeft;
      printf("Sending %d bytes of data\n", bytes);
       
      Sendto(sendfd, ptr, bytes, 0, sasend, salen);

      numBytesLeft -= bytes;
      ptr += bytes;
  
      // send delay
      //sleep(.1);
    }
  
   	
    //end = (double)times(&dummy)*ci;
  
    //
    // Send empty data to mark end of transmission
    //
    printf("Sending 0 bytes of data - end of data transmission\n");
    // send 3 times in case packets are dropped
    Sendto(sendfd, ptr, 0, 0, sasend, salen);
    //Sendto(sendfd, ptr, 0, 0, sasend, salen);	
    //Sendto(sendfd, ptr, 0, 0, sasend, salen);  

    // calculate time, print out information
    //end = (double)times(&dummy)*ci;
    //printf("Write took %0.3f seconds\n", end - start);

    printf("Data sent: %s", data);

    printf("Num bytes sent: %d\n", numBytes - numBytesLeft);

    int result;
    char buffer[ 1000 ];
    struct timeval _start,_end,_result;

    gettimeofday( &_start,NULL );
  
    // Writing is easy - it's just a quick call to sendto() 
    if ( ( result = sendto( sock, data, numBytes, 0, (struct sockaddr *) &addr,
			  sizeof(addr) ) ) < 0 ) {
    

      snprintf( buffer, 1000,
	      "Only wrote %d/%d bytes on multicast connection",
	      result, numBytes );
    
      Log::log( ERROR, buffer );
    }
    gettimeofday( &_end, NULL );
    timersub( &_end, &_start, &_result );
    snprintf( buffer, 1000, "Done writing %d multicast bytes in %0.3f ms. ",
	    numBytes, 
	    (float)_result.tv_sec*1000.0 + (float)_result.tv_usec/1000.0 );
    Log::log( DEBUG,  buffer );

  }
  else{
    Log::log( ERROR,
	      "Unrecognized multicast transfer mode" );
  }
  */
  
  cerr << "End of MulticastConnection::write, thread id is " << pthread_self() << endl;  
  return numBytes;
    
}

Connection **
MulticastConnection::getReadyToRead() {
  cerr << "In MulticastConnection::getReadyToRead, thread id is " << pthread_self() << endl;  
  cerr << "End of MulticastConnection::getReadyToRead, thread id is " << pthread_self() << endl;  
  /* This makes no sense here... */
  return NULL;
}

bool
MulticastConnection::isEqual( const Connection& c) {
  cerr << "In MulticastConnection::isEqual, thread id is " << pthread_self() << endl;  
  cerr << "End of MulticastConnection::isEqual, thread id is " << pthread_self() << endl;  
  return false;
}

void
MulticastConnection::close() {
  cerr << "In MulticastConnection::close, thread id is " << pthread_self() << endl;  

  /*
  if(SV_TRANSFER_MODE == 1){
    //
    // Reliable Multicasting Code
    //

    // This has been removed
  }
  else if(SV_TRANSFER_MODE == 2){
    //
    // Unreliable, Low-Level Multicasting Code - broken
    //

    //if ( theSocket != NO_SOCKET )
    //  Disconnect( &theSocket );


    if(sendfd != -1){
      Close(sendfd);
    }
  }
  else{
    Log::log( ERROR,
	      "Unrecognized multicast transfer mode" );
  }
  */
  cerr << "End of MulticastConnection::close, thread id is " << pthread_self() << endl;  
 
}
  
}
}
