
#include "listener.h"


int openListener(int port)
{
  int sockfd;
  struct sockaddr_in my_addr;    /* my address information */
  
  if ((sockfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)) == -1) {
    perror("socket");
    exit(1);
  }
  
  my_addr.sin_family = AF_INET;         /* host byte order */
  my_addr.sin_port = htons(port);       /* short, network byte order */
  my_addr.sin_addr.s_addr = INADDR_ANY; /* auto-fill with my IP */
  bzero(&(my_addr.sin_zero), 8);        /* zero the rest of the struct */
  
  if (bind(sockfd, (struct sockaddr *)&my_addr, 
	   sizeof(struct sockaddr)) == -1) 
  {
    perror("bind");
    exit(1);
  }
  
  return (sockfd);
}

int closeListener(int connfd)
{
  return close(connfd);
}














