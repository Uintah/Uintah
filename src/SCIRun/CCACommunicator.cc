/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  CCACommunicator.cc
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   June 2002
 *
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <strings.h>
#include <string.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>

#include <string>

#include "SCIRunFramework.h"
#include "CCACommunicator.h"

using namespace SCIRun;
using namespace std;


#define PORT  2009 // the port used for recv and send

//convert domain name to IP address
/** this method is not used
bool CCACommunicator::domainToIP(cosnt char *IP, const char *Addr)
{
  struct hostent *host=gethostbyname(Addr);
  if(host==NULL) return false;
  
  struct in_addr inaddr=*((struct in_addr *)host->h_addr);
  strcpy(IP, inet_ntoa(inaddr));
  return true;
}
*/

CCACommunicator::CCACommunicator(SCIRunFramework *framework,
				 const sci::cca::Services::pointer &svc)
{
  services=svc;
  ccaSiteList.push_back("qwerty.sci.utah.edu");
  //ccaSiteList.push_back("bugs.sci.utah.edu");
  ccaSiteList.push_back("rat.sci.utah.edu");  
  this->framework=framework;
}


void CCACommunicator::readPacket(const Packet &pkt)
{
  std::string url=pkt.ccaFrameworkURL;
  if(url==framework->getURL().getString()) return; //don't save my own URL
  for(unsigned int i=0; i<ccaFrameworkURL.size();i++){
    if(ccaFrameworkURL[i]==url) return;
  }
  //cerr<<"Received CCA Framework at "<<pkt.fromAddress<< endl;
  //cerr<<"URL="<<url<< endl;
  ccaFrameworkURL.push_back(url);

  
  sci::cca::ports::BuilderService::pointer bs = pidl_cast<sci::cca::ports::BuilderService::pointer>
    (services->getPort("cca.BuilderService"));
  if(bs.isNull()){
    cerr << "Fatal Error: Cannot find builder service\n";
    Thread::exitAll(1);
  }
  //do not delete the following line	
  //bs->registerFramework(url);

  services->releasePort("cca.BuilderService");
}

Packet CCACommunicator::makePacket(int i)
{
  char host[100];
  gethostname(host,99);
  Packet pkt;
  strcpy(pkt.fromAddress,host);
  strcpy(pkt.ccaFrameworkURL, framework->getURL().getString().c_str());
  return pkt;
}

void CCACommunicator::run()
{
 
  // connector's address information 
  struct sockaddr_in their_addr; 
  int sockfd;
  struct sockaddr_in myaddr;
  // Open a UDP socket. 
  sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd < 0){
    perror("error in socket: ");
    return;
  }
  // Bind the address to the socket. 
  bzero(&myaddr, sizeof(myaddr));
  myaddr.sin_family = AF_INET;
  myaddr.sin_addr.s_addr = htonl(INADDR_ANY);
  myaddr.sin_port = htons(PORT);
  if (bind(sockfd, (struct sockaddr *) &myaddr,sizeof(myaddr)) != 0){
    perror("error in bind:");
    return;
  }
 
  while(true){
    //Check incoming packages
    
    struct timeval tv;
    tv.tv_sec=10;    //timeout=1 seconds
    tv.tv_usec=0;   
    fd_set readfds;
    FD_ZERO(&readfds);              //empty the readfds
    FD_SET(sockfd,&readfds);        //add sockfd into readfds
    //select the file descriptors who are ready
    
    bool reading=true;
    while(reading){
      reading=false;
      int retval=select(sockfd+1, &readfds, NULL, NULL, &tv);
      if(retval==-1){
	perror("error in select:");
	return;
      }
      else if(retval>0){
      //check if sockfd is ready
	if(FD_ISSET(sockfd, &readfds)){
	  Packet pkt;
	  socklen_t addrlen=sizeof(sockaddr);
	  sockaddr addr;
	  if(recvfrom(sockfd, &pkt, sizeof(struct Packet), 0, &addr, &addrlen)!=0){
	    readPacket(pkt);
	    reading=true;
	  }
	}
      }
    }
    //Sending outging packages
    for(unsigned int i=0; i<ccaSiteList.size();i++){
      Packet pkt=makePacket(i);
      struct hostent *he;
      const char * destAddress=ccaSiteList[i].c_str();
      if ((he=gethostbyname(destAddress)) == NULL){ 
	cerr<<"Warning: unknown CCA site: "<<destAddress<<endl;
	continue;
      }
      their_addr.sin_family = AF_INET;         // host byte order 
      their_addr.sin_port = htons(PORT); // short, network byte order 
      their_addr.sin_addr = *((struct in_addr *)he->h_addr);
      bzero(&(their_addr.sin_zero), 8);        // zero the rest of the struct 
      
      int addrlen=sizeof(sockaddr_in);
      int retval=sendto(sockfd, &pkt, sizeof(struct Packet),0, 
		    (struct sockaddr *)&their_addr, addrlen);
      if(retval==-1) perror("error in send:");
      //else cerr<<"My Framework URL is sent to "<<destAddress<<endl;
    }
     
  }
    //need to terminate the above loop with Framework termination signal
    close(sockfd);
}
 
