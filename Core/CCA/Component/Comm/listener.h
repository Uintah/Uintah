
#ifndef _listener_H_
#define _listener_H_

#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netdb.h>
#include <sys/socket.h>

#ifdef __cplusplus  
extern "C" {
#endif  /* __cplusplus */   

/*
int openListener(const char * host,int port);

Starts listening to a socket 

port -- port to listen to

returns a file descriptor for the connection
*/
int openListener(int port);


/*
int closeListener(int connfd);

Stops the listening.

returns non-zero if an error has occured
*/
int closeListener(int connfd);

#ifdef __cplusplus  
}
#endif  /* __cplusplus */   

#endif

