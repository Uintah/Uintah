/*
	header to help with my socket stuff
	mostly all modified or taken from 
	UNIX Network Programming, Networking	APIs: Sockets and XTI
	Volume 1, second edition, by W. Richard Stevens 
	modifed/used by Oscar Barney
*/

#include	<sys/types.h>	/* basic system data types */
#include	<sys/socket.h>	/* basic socket definitions */
#include	<netinet/in.h>	/* sockaddr_in{} and other Internet defns */
#include	<arpa/inet.h>	/* inet(3) functions */
#include	<errno.h>
#include <iostream>
#include	<stdlib.h>
#include	<string.h>
//#include	<sys/uio.h>		/* for iovec{} and readv/writev */
#include	<unistd.h>
//#include	<sys/un.h>		/* for Unix domain sockets */

#define LISTENQ 1024				/* 2nd argument to listen() */
#define SERV_PORT 9877		/* TCP and UDP client-servers */
#define MAXLINE 4096			/* max text line length */
/* Following shortens all the type casts of pointer arguments */
#define	SA	struct sockaddr

using namespace std;

void print_error(string msg);

	/* prototypes for our socket wrapper functions: see {Sec errors} */
int Accept(int fd, struct sockaddr *sa, socklen_t *salenptr);
void Bind(int fd, const struct sockaddr *sa, socklen_t salen);
void Connect(int fd, const struct sockaddr *sa, socklen_t salen);
void Inet_pton(int family, const char *strptr, void *addrptr);
void Listen(int fd, int backlog);
void Setsockopt(int fd, int level, int optname, const void *optval, socklen_t *optlen);
int Socket(int family, int type, int protocol);

static ssize_t my_read(int fd, char *ptr);
ssize_t readline(int fd, void *vptr, size_t maxlen);
ssize_t Readline(int fd, void *ptr, size_t maxlen);
void startUpServer(int *listenfd, struct sockaddr_in *serverAddr);
ssize_t writen(int fd, const void *vptr, size_t n);
void Writen(int fd, void *ptr, size_t nbytes);

void Close(int fd);

//TODO fix waring that this is not used
//TODO test the goto more but its probably ok
//TODO may not be thread safe so fix that too
//figure 23.11 in Stevens talks about this problem.
static ssize_t my_read(int fd, char *ptr)
{
	static int  read_cnt = 0;
	static char *read_ptr;
	static char read_buf[MAXLINE];
	/*old way  TODO test new way more
	if (read_cnt <= 0) {
again:
	if ( (read_cnt = read(fd, read_buf, sizeof(read_buf))) < 0) {
	if (errno == EINTR)
	goto again;
	return(-1);
} else if (read_cnt == 0)
	return(0);
	read_ptr = read_buf;
}
	*/	
	int again = 0;
	if (read_cnt <= 0) {
		do{
			again = 0;
			if ( (read_cnt = read(fd, read_buf, sizeof(read_buf))) < 0) {
				if (errno == EINTR)
					again = 1;
				else
					return(-1);
			}
		}while(again == 1);	
		
		if (read_cnt == 0)
			return(0);
		read_ptr = read_buf;
	}

	read_cnt--;
	*ptr = *read_ptr++;
	return(1);
}
