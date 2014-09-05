
#ifndef _connector_H_
#define _connector_H_

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
int open_connection(const char * host,int port);

Opens a socket and a connection to a remote computer

host -- The name of the remote computer or its IP address
port -- port to open connection to

returns a file descriptor for the connection
*/
int open_connection(const char * host,int port);


/*
int close_connection(int connfd);

closes the connection.

returns non-zero if an error has occured
*/
int close_connection(int connfd);

#ifdef __cplusplus  
}
#endif  /* __cplusplus */   

#endif
