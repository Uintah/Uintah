/*
 *
 * CLIENT
 *
 * MulticastConnection: Provides a multicast network connection.
 * This class is totally broken -- don't use it!
 * 
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
//#include <Network/unpv12e/mcast/unp.h>

namespace SemotusVisum {

MulticastConnection::MulticastConnection(){
  //ur = NULL;
  cerr << "In MulticastConnection::MulticastConnection, thread id = " << pthread_self() << endl;
  cerr << "End of MulticastConnection::MulticastConnection, thread id = " << pthread_self() << endl;
}


MulticastConnection::MulticastConnection( const char * group,
					  const int port ) {
   Log::log( ENTER, "[MulticastConnection::MulticastConnection] entered, thread id = " + mkString((int) pthread_self()) );
  /* Initialize the connection */

  /*
  if(SV_TRANSFER_MODE == 1){

    //
    // Reliable Multicasting Code
    //

    // This code has been removed

  }
  else if(SV_TRANSFER_MODE == 2){
  
    //
    // Unreliable, Low-Level Multicasting Code
    //

    socklen_t			salen;
    struct sockaddr		*sasend;
    const int			on = 1;

    // reconstruct sender's socket address
    Udp_client("228.6.6.6", "6210", (void **) &sasend, &salen);

    recvfd = Socket(sasend->sa_family, SOCK_DGRAM, 0);

    Setsockopt(recvfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

    sarecv = (sockaddr *)Malloc(salen);
    memcpy(sarecv, sasend, salen);
    Bind(recvfd, sarecv, salen);

    Mcast_join(recvfd, sasend, salen, NULL, 0);

  // create what looks like an ordinary UDP socket 
  if ((sock=socket(AF_INET,SOCK_DGRAM,0)) < 0) {
    Log::log( ERROR, string("Error creating multicast socket: ") +
	      strerror( errno ));
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

  // My added code

  int n = sizeof(struct sockaddr);
  if ( bind(sock, (struct sockaddr *) &addr, n ) != 0 ){
    Log::log( ERROR,
	      "Error binding multicast socket" );
    Log::log( ERROR, strerror( errno ) );
    cerr << "Bind " << strerror( errno ) << endl;  
    return;
  }
  // We're ready to go... 


  }
  else{
    Log::log( ERROR, "[MulticastConnection::MulticastConnection] Unrecognized multicast data transfer mode" );

  }
  */
   
  Log::log( LEAVE, "[MulticastConnection::MulticastConnection] leaving, thread id = " + mkString((int) pthread_self()) );
}

  
MulticastConnection::~MulticastConnection() {
  Log::log( ENTER, "[MulticastConnection::~MulticastConnection] entered, thread id = " + mkString((int) pthread_self()) );

  /*
  if(SV_TRANSFER_MODE == 1){

    //
    // Reliable Multicasting Code
    //

    // This code has been removed
  }
  else if(SV_TRANSFER_MODE == 2){ 
    //
    // Unreliable, Low-Level Multicasting Code - broken
    //

     
    if(recvfd != -1){
      Close(recvfd);
    }
    
  }
  else{
    Log::log( ERROR, "[MulticastConnection::~MulticastConnection] Unrecognized multicast data transfer mode" );
  }
  */
  Log::log( LEAVE, "[MulticastConnection::~MulticastConnection] leaving, thread id = " + mkString((int) pthread_self()) );
}

  
int
MulticastConnection::read( char * data, int numBytes ) {
  Log::log( ENTER, "[MulticastConnection::read] entered, thread id = " + mkString((int) pthread_self()) );

   int length = 0;
  /*
  if(SV_TRANSFER_MODE == 1){
    //
    // Reliable Multicasting Code
    //

    // This code has been removed
  }
  else if(SV_TRANSFER_MODE == 2){

   

    //
    // Unreliable, Low-Level Multicasting Code
    //
  
  	unsigned int			n, numBytesLeft, numPackets, packetsReceived;
	//char				line[MAXLINE+1];
        //char				data[maxNumBytes+1];
        
        const char * ptr;
	socklen_t			len;
	struct sockaddr		*safrom;

	safrom = (sockaddr *) Malloc(salen); // FIXME -- use scinew instead of Malloc

        packetsReceived = 0;

        len = salen;
        printf("Receiving...\n");
	
          //printf("Receiving...\n");
	  
          // 
          // Basic receive code
          //

   
          //
          // Receive data
          //
 
          numPackets = (int) ((((float)numBytes)/MAXLINE) + 1);
          packetsReceived = 0;
          numBytesLeft = numBytes;

          data = (char *)Malloc(numBytes); // FIXME -- use scinew instead of Malloc
          ptr = data;

          // while we either have bytes left receive or we haven't 
          // received a terminating packet
	  while(numBytesLeft > 0 && n != 0){
            
            n = Recvfrom(recvfd, (void *)ptr, MAXLINE, 0, safrom, &len);
            numBytesLeft -= n;
         
            if(n != 0){
              ptr += n;
              packetsReceived += 1;
            }

	    printf("Received %d bytes from %s\n", n, Sock_ntop(safrom, len));
            printf("Number of bytes left: %d\n", numBytesLeft);
          }

      	  data[numBytes] = 0;	// null terminated
          printf("Received %d total bytes from %s\n", numBytes - numBytesLeft, Sock_ntop(safrom, len));
          printf("Data received: %s\n", data); 
          printf("Received %d/%d packets -- %0.0f\% reliability\n\n", packetsReceived, numPackets, ((float) packetsReceived/numPackets)*100);
 
	  length = numBytes - numBytesLeft;


  }
  else{
    Log::log( ERROR, "[MulticastConnection::read] Unrecognized multicast data transfer mode" );
  }
  */

  Log::log( LEAVE, "[MulticastConnection::read] leaving, thread id = " + mkString((int) pthread_self()) );
  return length;

  
}

int
MulticastConnection::receiveData( char * data, int numBytes ) {
  Log::log( ENTER, "[MulticastConnection::receiveData] entered, thread id = " + mkString((int) pthread_self()) );

  // This code has been removed

  Log::log( LEAVE, "[MulticastConnection::receiveData] leaving, thread id = " + mkString((int) pthread_self()) );

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

  Log::log( ENTER, "[MulticastConnection::write] entered, thread id = " + mkString((int) pthread_self()) );

  // not sure how this is supposed to be used...

  //
  // Unreliable, Low-Level Multicasting Code
  //

  /*
  int result;
  struct timeval _start,_end,_result;

  gettimeofday( &_start,NULL );
  
  // Writing is easy - it's just a quick call to sendto() 
  if ( ( result = sendto( sock, data, numBytes, 0, (struct sockaddr *) &addr,
			  sizeof(addr) ) ) < 0 ) {
    
    Log::log( ERROR, string("Only wrote ") + mkString(result) + "/" +
	      mkString(numBytes) + " bytes on multicast connection" );
  }
  
  gettimeofday( &_end, NULL );
  timersub( &_end, &_start, &_result );
  Log::log( DEBUG, string("Done writing ") + mkString(numBytes) +
    " multicast bytes in " + mkString((float)_result.tv_sec*1000.0 +
					      (float)_result.tv_usec/1000.0) +
	    " ms. ");

  cerr << "End of MulticastConnection::write, thread id = " << pthread_self() << endl;
  return result;
  */
  Log::log( LEAVE, "[MulticastConnection::write] leaving, thread id = " + mkString((int) pthread_self()) );
  return 0;
}

bool
MulticastConnection::isEqual( const Connection& c) {
  Log::log( ENTER, "[MulticastConnection::isEqual] entered, thread id = " + mkString((int) pthread_self()) );
  Log::log( LEAVE, "[MulticastConnection::isEqual] leaving, thread id = " + mkString((int) pthread_self()) );
  return false;
}

void
MulticastConnection::close() {
  Log::log( ENTER, "[MulticastConnection::close] entered, thread id = " + mkString((int) pthread_self()) );

  /*
  if(SV_TRANSFER_MODE == 1){
    
    //
    // Reliable Multicasting Code
    //

    // This code has been removed

  }
  else if(SV_TRANSFER_MODE == 2){

    //
    // Unreliable, Low-Level Multicasting Code
    //

    //if ( theSocket != NO_SOCKET )
    //  Disconnect( &theSocket );

    if(recvfd != -1){
      Close(recvfd);
    }
  }
  else{
    Log::log( ERROR,
	      "[MulticastConnection::close] Unrecognized multicast data transfer mode" );
  }
  */
  
  Log::log( LEAVE, "[MulticastConnection::close] leaving, thread id = " + mkString((int) pthread_self()) );
  
}
 
}






