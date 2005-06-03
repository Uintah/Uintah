/* $Id$ */


/*
 * Copyright © 2000 The Regents of the University of California. 
 * All Rights Reserved. 
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for educational, research and non-profit purposes, without
 * fee, and without a written agreement is hereby granted, provided that the
 * above copyright notice, this paragraph and the following three paragraphs
 * appear in all copies. 
 *
 * Permission to incorporate this software into commercial products may be
 * obtained by contacting
 * Eric Lund
 * Technology Transfer Office 
 * 9500 Gilman Drive 
 * 411 University Center 
 * University of California 
 * La Jolla, CA 92093-0093
 * (858) 534-0175
 * ericlund@ucsd.edu
 *
 * This software program and documentation are copyrighted by The Regents of
 * the University of California. The software program and documentation are
 * supplied "as is", without any accompanying services from The Regents. The
 * Regents does not warrant that the operation of the program will be
 * uninterrupted or error-free. The end-user understands that the program was
 * developed for research purposes and is advised not to rely exclusively on
 * the program for any reason. 
 *
 * IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
 * LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION,
 * EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE. THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED
 * HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO
 * OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
 * MODIFICATIONS. 
 */


#include "config.h"
#include <sys/types.h>   /* sometimes required for #include <sys/socket.h> */
#include <sys/socket.h>  /* AF_INET */
#include <netinet/in.h>  /* IPPROTO_TCP struct in_addr */
#include <arpa/inet.h>   /* inet_addr inet_ntoa */
#include <netdb.h>       /* gethostby{addr,name} */
#include <sys/time.h>    /* struct timeval */
#include <stdlib.h>      /* realloc */
#include <string.h>      /* memcpy memset strchr strcasecmp strncpy */
#include <unistd.h>      /* close read write */
#define ASIP_SHORT_NAMES
#include "ipseed.h"

#include <sys/ioctl.h>
#include <stdio.h>
/*
 * Note: some systems take (char *) as the third parameter to [gs]etsockopt,
 * others (void *).  Since the former is compatable with the latter, we always
 * cast to (char *).
 */


/* Value for socket call values if not in system include. */
#ifndef MSG_WAITALL
#  define MSG_WAITALL 0x100
#endif
#ifndef TCP_NODELAY
#  define TCP_NODELAY 0x01
#endif

/* Convenient shorthand for casting. */
typedef struct sockaddr *SAP;

/*
 * We cache hostents locally to avoid going to the DNS too often.  This also
 * gets around an old Solaris bug which leaks memory whenever dlopen is called,
 * such as on the dynamic DNS lookup library.
 */
static struct hostent *cache = NULL;
static unsigned int cacheCount = 0;


static void **
ListCopy(const void **list,
         size_t elementSize);
static void
ListFree(void **list);


/*
 * Looks in the name and alias entries of #hostEntry# for a fully-qualified
 * name.  Returns the fqn if found; otherwise, returns the name entry.
 */
static const char *
BestHostName(const struct hostent *hostEntry) {
  int i;
  if(strchr(hostEntry->h_name, '.') == NULL) {
    for(i = 0; hostEntry->h_aliases[i] != NULL; i++) {
      if(strchr(hostEntry->h_aliases[i], '.') != NULL)
        return hostEntry->h_aliases[i]; /* found! */
    }
  }
  return hostEntry->h_name;
}


/*
 * Appends a copy of #hostEntry# to the global map cache.  Returns a pointer to
 * the copy, or NULL on error.
 */
static struct hostent *
CacheHostent(const struct hostent *hostEntry) {

  struct hostent *extendedCache;
  struct hostent newEntry;

  extendedCache = (struct hostent *)
    realloc(cache, (cacheCount + 1) * sizeof(struct hostent));
  if(extendedCache == NULL)
    return NULL; /* Out of memory. */
  cache = extendedCache;

  newEntry.h_addrtype = hostEntry->h_addrtype;
  newEntry.h_length = hostEntry->h_length;

  if((newEntry.h_name = strdup(hostEntry->h_name)) == NULL)
    return NULL; /* Out of memory. */

  if((newEntry.h_aliases = (char **)
        ListCopy((const void **)hostEntry->h_aliases, 0)) == NULL) {
    free((char *)newEntry.h_name);
    return NULL; /* Out of memory. */
  }

  if((newEntry.h_addr_list = (char **)
        ListCopy((const void **)hostEntry->h_addr_list, newEntry.h_length)) ==
      NULL) {
    ListFree((void **)newEntry.h_aliases);
    free((char *)newEntry.h_name);
    return NULL; /* Out of memory. */
  }

  cache[cacheCount] = newEntry;
  return &cache[cacheCount++];

}


/* Initializes #addr# to refer to #address#:#port#. */
static void
InitSockaddrIn(struct sockaddr_in *addr,
               Address address,
               Port port) {
  memset(addr, 0, sizeof(struct sockaddr_in));
  addr->sin_addr.s_addr = (address == ANY_ADDRESS) ? INADDR_ANY : address;
  addr->sin_family = AF_INET;
  addr->sin_port = htons((port == ANY_PORT) ? 0 : port);
}


/*
 * Returns a NULL-terminated list containing a copy of every item in the NULL-
 * terminated list of pointers #list#.  #elementSize# specifies the size of
 * the data items pointed to by the list elements; a value of 0 indicates that
 * the list elements are char *.
 */
static void **
ListCopy(const void **list,
         size_t elementSize) {

  int i;
  unsigned int listLen;
  void **returnValue;

  for(listLen = 0; list[listLen] != NULL; listLen++)
    ; /* Nothing more to do. */

  if((returnValue = (void **)malloc((listLen + 1) * sizeof(void *))) == NULL)
    return NULL; /* Out of memory. */

  for(i = 0; i < listLen; i++) {
    if(elementSize == 0) {
      returnValue[i] = strdup((char *)list[i]);
    }
    else {
      returnValue[i] = malloc(elementSize);
      if(returnValue[i] != NULL) {
        memcpy(returnValue[i], list[i], elementSize);
      }
    }
    if(returnValue[i] == NULL) {
      ListFree(returnValue);
      return NULL; /* Out of memory. */
    }
  }
  returnValue[listLen] = NULL;

  return returnValue;

}


/*
 * Frees every element of the NULL-terminated list of pointers #list#, along
 * with #list# itself.
 */
static void
ListFree(void **list) {
  for(; *list != NULL; list++)
    free(*list);
  free(list);
}


/*
 * Searches the DNS mapping cache for #address#, adding a new entry if needed.
 * Returns a pointer to the mapping entry, or NULL on error.
 */
static const struct hostent*
LookupByAddress(Address address) {

  struct in_addr addrAsInAddr;
  struct hostent *addrEntry;
  struct in_addr **cachedAddr;
  int i;

  for(i = 0; i < cacheCount; i++) {
    for(cachedAddr = (struct in_addr**)cache[i].h_addr_list;
        *cachedAddr != NULL;
        cachedAddr++) {
      if((**cachedAddr).s_addr == address)
        return &cache[i];
    }
  }

  addrAsInAddr.s_addr = address;
  addrEntry =
    gethostbyaddr((char *)&addrAsInAddr, sizeof(addrAsInAddr), AF_INET);
  if(addrEntry == NULL)
    return NULL;
  else if(addrEntry->h_length != sizeof(struct in_addr))
    return NULL; /* We don't (yet) handle IPv6 addresses. */

  addrEntry = CacheHostent(addrEntry);
  return addrEntry;

}


/*
 * Searches the DNS mapping cache for #name#, adding a new entry if needed.
 * Returns a pointer to the mapping entry, or NULL on error.
 */
static const struct hostent*
LookupByName(const char *name) {

  char **cachedName;
  char **extendedAliases;
  struct hostent *nameEntry;
  int i;
  int listLen;

  for(i = 0; i < cacheCount; i++) {
    if(strcasecmp(name, cache[i].h_name) == 0)
      return &cache[i];
    for(cachedName = cache[i].h_aliases; *cachedName != NULL; cachedName++) {
      if(strcasecmp(*cachedName, name) == 0)
        return &cache[i];
    }
  }

  nameEntry = gethostbyname(name);
  if(nameEntry == NULL)
    return NULL;
  else if(nameEntry->h_length != sizeof(struct in_addr))
    return NULL; /* We don't (yet) handle IPv6 addresses. */

  /* We extend cached entries' h_aliases lists to include nicknames. */
  for(i = 0; i < cacheCount; i++) {
    if(strcmp(nameEntry->h_name, cache[i].h_name) == 0) {
      for(listLen = 0; cache[i].h_aliases[listLen] != NULL; listLen++)
        ; /* Nothing more to do. */
      extendedAliases =
        (char **)realloc(cache[i].h_aliases, sizeof(char **) * (listLen + 2));
      if(extendedAliases != NULL) {
        extendedAliases[listLen] = strdup(name);
        extendedAliases[listLen + 1] = NULL;
        cache[i].h_aliases = extendedAliases;
      }
      return &cache[i];
    }
  }

  nameEntry = CacheHostent(nameEntry);
  return nameEntry;

}


static int
ReceiveBytes(Socket sock,
             void *intoWhere,
             size_t size,
             Address *addr,
             Port *port) {
  struct sockaddr_in peer;
  ASIP_SOCKLEN_T peerLen = sizeof(peer);
  int received;
  received = recvfrom(sock, intoWhere, size, MSG_WAITALL, (SAP)&peer, &peerLen); 
  if(addr != NULL)
    *addr = peer.sin_addr.s_addr;
  if(port != NULL)
    *port = ntohs(peer.sin_port);
  return received;
}


Socket
Accept(Socket sock) {
  struct sockaddr_in peer;
  ASIP_SOCKLEN_T peerLen = sizeof(peer);
  return accept(sock, (SAP)&peer, &peerLen);
}


const char *
AddressImage(Address addr) {
  struct in_addr addrAsInAddr;
  addrAsInAddr.s_addr = addr;
  return inet_ntoa(addrAsInAddr);
}


const char *
AddressMachine(Address addr) {
  const struct hostent *hostEntry;
  static char returnValue[63 + 1];
  hostEntry = LookupByAddress(addr);
  strncpy(returnValue,
          (hostEntry == NULL) ? "" : BestHostName(hostEntry),
          sizeof(returnValue));
  return returnValue;
}


int
AddressValues(const char *machineOrAddress,
              Address *addressList,
              unsigned int atMost) {

  const struct hostent *hostEntry;
  int i;
  int itsAnAddress;

  /*
   * inet_addr() has the weird behavior of returning an unsigned quantity but
   * using -1 as an error value.  Furthermore, the value returned is sometimes
   * int and sometimes long, complicating the test.  Once inet_aton() is more
   * widely available, we should switch to using it instead.
   */
  itsAnAddress = (inet_addr(machineOrAddress) ^ -1) != 0;

  if(itsAnAddress && (atMost == 1)) {
    *addressList = inet_addr(machineOrAddress);
    return 1;
  }

  hostEntry = itsAnAddress ?
              LookupByAddress(inet_addr(machineOrAddress)) :
              LookupByName(machineOrAddress);
  if(hostEntry == NULL) {
    return 0;
  }
  else if(atMost == 0) {
    return 1;
  }

  for(i = 0; i < atMost; i++) {
    if(hostEntry->h_addr_list[i] == NULL) {
      break;
    }
    addressList[i] = ((struct in_addr **)hostEntry->h_addr_list)[i]->s_addr;
  }

  return i;

}


int
CheckIfAnyReadable(Socket *socks,
                   Socket *readable,
                   double secondsToWait) {

  int i;
  Socket maxSock;
  fd_set readFDs;
  int returnValue;
  struct timeval timeOut;
  struct timeval *waitTime;

  FD_ZERO(&readFDs);
  maxSock = 0;
  for(; *socks != NO_SOCKET; socks++) {
    FD_SET(*socks, &readFDs);
    if(*socks > maxSock)
      maxSock = *socks;
  }

  if(secondsToWait == WAIT_FOREVER) {
    waitTime = NULL;
  }
  else {
    timeOut.tv_sec  = (int)secondsToWait;
    timeOut.tv_usec = (int)(secondsToWait - timeOut.tv_sec) * 1000000;
    waitTime = &timeOut;
  }
  returnValue = select(maxSock + 1, &readFDs, NULL, NULL, waitTime) > 0;

  if(returnValue > 0 && readable != NULL) {
    for(i = 0; i <= maxSock; i++) {
      if(FD_ISSET(i, &readFDs))
        *readable++ = i;
    }
    *readable = NO_SOCKET;
  }

  return returnValue;

}


int
CheckIfReadable(Socket sock,
                double secondsToWait) {
  Socket twoSockets[2];
  twoSockets[0] = sock;
  twoSockets[1] = NO_SOCKET;
  return CheckIfAnyReadable(twoSockets, NULL, secondsToWait);
}


Socket
ConnectToIpPortBuffered(Protocols protocol,
                        Address address,
                        Port port,
                        size_t receiveBufferSize,
                        size_t sendBufferSize) {

  struct sockaddr_in addrAndPort;
  int intSize;
  Socket sock;
  int turnOn = 1;

  if(protocol == TCP_PROTOCOL) {
    if((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
      return NO_SOCKET;
    (void)setsockopt(sock, IPPROTO_TCP, TCP_NODELAY,
                     (char *)&turnOn, sizeof(turnOn));
  }
  else if((sock = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    return NO_SOCKET;
  }

  if(receiveBufferSize != DEFAULT_BUFFER_SIZE) {
    intSize = receiveBufferSize;
    (void)setsockopt(sock, SOL_SOCKET, SO_RCVBUF,
                     (char *)&intSize, sizeof(intSize));
  }

  if(sendBufferSize != DEFAULT_BUFFER_SIZE) {
    intSize = sendBufferSize;
    (void)setsockopt(sock, SOL_SOCKET, SO_SNDBUF,
                     (char *)&intSize, sizeof(intSize));
  }

  InitSockaddrIn(&addrAndPort, address, port);

  if(connect(sock, (SAP)&addrAndPort, sizeof(addrAndPort)) < 0) {
    Disconnect(&sock);
    return NO_SOCKET;
  }

  return sock;

}


void
Disconnect(Socket *sock) {
  /*
  Socket s = *sock;
  if(s == NO_SOCKET)
    return;
  *sock = NO_SOCKET;
  shutdown(s, 2); // ejl
  (void)close(s);*/

  if ( *sock == NO_SOCKET ) {
    printf("Closing a no socket");
    return;
  }
  shutdown( *sock, 2 );
  close( *sock );
  *sock = NO_SOCKET;
    
}


const char *
MyMachineName(void) {

  extern int
  gethostname();  /* Doesn't always seem to be in a system include file. */

  const struct hostent *myEntry;
  static char returnValue[100] = "";

  /* If we have a value in returnValue, then we've already done the work. */
  if(returnValue[0] == '\0') {
    /* try the simple case first */
    if(gethostname(returnValue, sizeof(returnValue)) < 0)
      return 0;
    if(!strchr(returnValue, '.')) {
      /* Okay, that didn't work; take what we can get from the DNS. */
      myEntry = LookupByName(returnValue);
      if(myEntry == NULL)
        return NULL;
      strncpy(returnValue, BestHostName(myEntry), sizeof(returnValue));
      returnValue[sizeof(returnValue) - 1] = '\0';
    }
  }

  return returnValue;

}


Socket
OpenIpPortBuffered(Protocols protocol,
                   Address address,
                   Port port,
                   size_t receiveBufferSize,
                   size_t sendBufferSize,
                   Port *openedPort) {

  struct sockaddr_in addrAndPort;
  ASIP_SOCKLEN_T addrAndPortLen = sizeof(addrAndPort);
  int intSize;
  Socket sock;
  int turnOn = 1;

  
  if(protocol == TCP_PROTOCOL) {
    if((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
      return NO_SOCKET;
    (void)setsockopt(sock, IPPROTO_TCP, TCP_NODELAY,
                     (char *)&turnOn, sizeof(turnOn));
  }
  else if((sock = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    return NO_SOCKET;
  }

  (void)setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
                   (char *)&turnOn, sizeof(turnOn));

  if(receiveBufferSize != DEFAULT_BUFFER_SIZE) {
    intSize = receiveBufferSize;
    (void)setsockopt(sock, SOL_SOCKET, SO_RCVBUF,
                     (char *)&intSize, sizeof(intSize));
  }

  if(sendBufferSize != DEFAULT_BUFFER_SIZE) {
    intSize = sendBufferSize;
    (void)setsockopt(sock, SOL_SOCKET, SO_SNDBUF,
                     (char *)&intSize, sizeof(intSize));
  }

  InitSockaddrIn(&addrAndPort, address, port);

  if(bind(sock, (SAP)&addrAndPort, addrAndPortLen) < 0) {
    Disconnect(&sock);
    return NO_SOCKET;
  }

  /*
   * Note: According to Stevens' book, there's no obvious best number for the
   * second parameter to listen, but 15 is reasonable.
   */
  if(protocol == TCP_PROTOCOL && listen(sock, 15) < 0) {
    Disconnect(&sock);
    return NO_SOCKET;
  }

  if(openedPort != NULL &&
     getsockname(sock, (SAP)&addrAndPort, &addrAndPortLen) >= 0)
    *openedPort = ntohs(addrAndPort.sin_port);

  return sock;

}


Address
PeerAddress(Socket sock) {
  struct sockaddr_in addrAndPort;
  ASIP_SOCKLEN_T addrAndPortLen = sizeof(addrAndPort);
  return (getpeername(sock, (SAP)&addrAndPort, &addrAndPortLen) < 0) ?
         ANY_ADDRESS : addrAndPort.sin_addr.s_addr;
}


Port
PeerPort(Socket sock) {
  struct sockaddr_in addrAndPort;
  ASIP_SOCKLEN_T addrAndPortLen = sizeof(addrAndPort);
  return (getpeername(sock, (SAP)&addrAndPort, &addrAndPortLen) < 0) ?
         ANY_PORT : ntohs(addrAndPort.sin_port);
}


int
ReceiveFromTerminated(Socket sock,
                      void *intoWhere,
                      size_t size,
                      const char *terminator,
                      Address *addr,
                      Port *port) {

  int received;

  if(terminator == NULL) {
    received = ReceiveBytes(sock, intoWhere, size, addr, port);
    /* OLD - WE NEED THE RETURN CODE!!!
       return (received < 0) ? 0 : received;
    */
    return received;
  }

  for(received = 0;
      received < size;
      received++, intoWhere = (char *)intoWhere + 1) {
    if(!ReceiveBytes(sock, intoWhere, 1, addr, port))
      return 0;
    if(*(char *)intoWhere == *terminator) {
      received++;
      break;
    }
  }

  return received;

}


int
SendTo(Socket sock,
       const void *fromWhere,
       size_t size,
       Address addr,
       Port port) {
  struct sockaddr_in peer;
  int written;
  static int i = 0;

  if ( !i ) {
    unsigned long bogus = 0;
    if ( ioctl(sock, FIONBIO, &bogus) < 0 )
      perror("\tERROR - ioctl");
    //i++;
  }
  
  if(addr != ANY_ADDRESS) {
    InitSockaddrIn(&peer, addr, port);
    return sendto(sock, fromWhere, size, 0, (SAP)&peer, sizeof(peer)) == size;
  }
  for( ;
      size > 0 && (written = write(sock, fromWhere, size)) > 0;
      size -= written, fromWhere = (char *)fromWhere + written)
    ; /* Nothing more to do. */
  
  /*return size == 0; OLD - we want the # of bytes written */
  return written;
}


Address
SocketAddress(Socket sock) {
  struct sockaddr_in addrAndPort;
  ASIP_SOCKLEN_T addrAndPortLen = sizeof(addrAndPort);
  return (getsockname(sock, (SAP)&addrAndPort, &addrAndPortLen) < 0) ?
         ANY_ADDRESS : addrAndPort.sin_addr.s_addr;
}


size_t
SocketBufferSize(Socket sock,
                      int sendSize) {
  size_t size;
  ASIP_SOCKLEN_T sizeLen = sizeof(size);
  return (getsockopt(sock,
                     SOL_SOCKET,
                     sendSize ? SO_SNDBUF : SO_RCVBUF,
                     (char *)&size,
                     &sizeLen) == 0) ? size : 0;
}


Port
SocketPort(Socket sock) {
  struct sockaddr_in addrAndPort;
  ASIP_SOCKLEN_T addrAndPortLen = sizeof(addrAndPort);
  return (getsockname(sock, (SAP)&addrAndPort, &addrAndPortLen) < 0) ?
         ANY_PORT : ntohs(addrAndPort.sin_port);
}
