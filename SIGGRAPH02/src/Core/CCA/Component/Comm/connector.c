
//#define DEBUG
#include "connector.h"

int open_connection(const char * host,int port)
{
  int connfd;
  struct sockaddr_in their_addr; /* connector's address information */
  struct hostent *he;
  
  if ((he=gethostbyname(host)) == NULL) {  /* get the host info */
#ifdef __DEBUG		
    herror("gethostbyname");
#endif
    return -1;
  }
  
  if ((connfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)) == -1) {
#ifdef __DEBUG		
    perror("socket");
#endif		
    return -1;
  }
  
  their_addr.sin_family = AF_INET;      /* host byte order */
  their_addr.sin_port = htons(port);  /* short, network byte order */
  their_addr.sin_addr = *((struct in_addr *)he->h_addr);
  bzero(&(their_addr.sin_zero), 8);     /* zero the rest of the struct */
  
  if ( connect( connfd, (struct sockaddr *)&their_addr,
		sizeof(their_addr) ) < 0 )
    {
      close( connfd );
#ifdef __DEBUG			
      perror("connect");
#endif	
      return -1;
    }
  return connfd;	
}

int close_connection(int connfd)
{
  return close(connfd);
}

