/*
	implementations of helper.h stuff
	mostly all modified or taken from 
	UNIX Network Programming, Networking	APIs: Sockets and XTI
	Volume 1, second edition, by W. Richard Stevens 
	modifed/used by Oscar Barney
*/

#include <Packages/Ptolemy/Core/Comm/NetworkHelper.h>

//TODO sometimes we do not want to print a message.
//what we want to do is kill scirun and let Ptolemy time out?
//or does the ssh actor send back the cerr messages?
//TODO optionally email back some error when
//we cannot send it over socket
void print_error(string msg)
{ 
	cerr << msg << " has occured." << endl;
}


int Accept(int fd, struct sockaddr *sa, socklen_t *salenptr)
{
	int n;
	do{
		if ( (n = accept(fd, sa, salenptr)) < 0) {
#ifdef	EPROTO
			if (errno == EPROTO || errno == ECONNABORTED)
#else
			if (errno == ECONNABORTED)
#endif
				n=-2;
		else
			print_error("accept error");
		}
	} while (n == -2);

	return(n);
}

void Bind(int fd, const struct sockaddr *sa, socklen_t salen)
{//TODO probably want to exit scirun if there is a bind error
	if (bind(fd, sa, salen) < 0)
		print_error("bind error");
}

void Connect(int fd, const struct sockaddr *sa, socklen_t salen)
{
	if (connect(fd, sa, salen) < 0)
		print_error("connect error");
}

void Inet_pton(int family, const char *strptr, void *addrptr)
{
	int n;
	if ( (n = inet_pton(family, strptr, addrptr)) < 0)
		print_error("inet_pton error for " + string(strptr));	/* errno set */
	else if (n == 0){
		print_error("inet_pton error for " + string(strptr) + ", quit");	/* errno not set, was err_quit*/
		exit(1);
	}
}

void Listen(int fd, int backlog)
{
	if (listen(fd, backlog) < 0)
		print_error("listen error");
}

int Socket(int family, int type, int protocol)
{
	int n;
	if ( (n = socket(family, type, protocol)) < 0)
		print_error("socket error");
	return(n);
}


//reading and writing
ssize_t readline(int fd, void *vptr, size_t maxlen)
{
	int n, rc;
	char c, *ptr;
	ptr = (char*)vptr;
	for (n = 1; n < (int)maxlen; n++) {
		if ( (rc = my_read(fd, &c)) == 1) {
			*ptr++ = c;
			if (c == '\n')
				break;	/* newline is stored, like fgets() */
		} else if (rc == 0) {
			if (n == 1)
				return(0);	/* EOF, no data read */
			else
				break;		/* EOF, some data was read */
		} else
			return(-1);		/* error, errno set by read() */
	}

	*ptr = 0;	/* null terminate like fgets() */
	return(n);
}

ssize_t Readline(int fd, void *ptr, size_t maxlen)
{
	ssize_t n;
	if ( (n = readline(fd, ptr, maxlen)) < 0)
		print_error("readline error");
	return(n);
}


//start up the server
void startUpServer(int *listenfd, struct sockaddr_in *serverAddr)
{
	*listenfd = Socket(AF_INET, SOCK_STREAM, 0);

	bzero(serverAddr, sizeof(*serverAddr));
	serverAddr->sin_family = AF_INET;
	serverAddr->sin_addr.s_addr = htonl(INADDR_ANY);
	serverAddr->sin_port = htons(SERV_PORT);

	Bind(*listenfd, (SA *) serverAddr, sizeof(*serverAddr));
	Listen(*listenfd, LISTENQ);	
}

						/* Write "n" bytes to a descriptor. */
ssize_t writen(int fd, const void *vptr, size_t n)
{
	size_t nleft;
	ssize_t nwritten;
	const char *ptr;

	ptr = (char*)vptr;
	nleft = n;
	while (nleft > 0) {
		if ( (nwritten = write(fd, ptr, nleft)) <= 0) {
			if (errno == EINTR)
				nwritten = 0;		/* and call write() again */
			else
				return(-1);			/* error */
		}

		nleft -= nwritten;
		ptr   += nwritten;
	}
	return(n);
}

void Writen(int fd, void *ptr, size_t nbytes)
{
	if (writen(fd, ptr, nbytes) != (ssize_t)nbytes)
		print_error("writen error");
}


//unix stuff
void Close(int fd)
{
	if (close(fd) == -1)
		print_error("close error");
}
 
