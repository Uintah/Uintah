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
#include <sys/time.h>   /* struct tv */
#include <stdio.h>      /* sprintf */
#include <stdlib.h>     /* free malloc */
#include <string.h>     /* strlen strncmp */
#include <unistd.h>     /* gettimeofday */
#define ASIP_SHORT_NAMES
#include "ipseed.h"
#define ASFMT_SHORT_NAMES
#include "formatseed.h"
#define ASNETRES_SHORT_NAMES
#include "netresourceseed.h"

#define LF   "\012"
#define CRLF "\015" LF
#ifndef NULL
#  define NULL 0
#endif

#define HTTP_HOST        "Host: %s" CRLF
#define HTTP_TRACE       "TRACE * HTTP/1.1" CRLF
#define NWS_TCP_BW_REQ    800
#define NWS_TCP_HANDSHAKE 411
#define NWS_TCP_REUSE     0
#define NWS_VERSION       0x02000000
#define SMTP_BODY_BEGIN   "DATA"
#define SMTP_BODY_END     CRLF "." CRLF
#define SMTP_HELLO        "HELO %s" CRLF
#define SMTP_NOOP         "NOOP" CRLF
#define SMTP_OK           "250"
#define SMTP_RECEIVER     "RCPT TO: <nobody@%s>" CRLF
#define SMTP_SENDER       "MAIL FROM: <nobody@%s>" CRLF


/* Returns the number of microseconds since the beginning of the epoch. */
static long
MicroTime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return(tv.tv_sec * 1000000 + tv.tv_usec);
}


/*
 * Sends #messageCount# messages, each #bytesPerMessage# bytes long, on #sock#,
 * followed by #terminator# if it is non-null.  Afterward, waits for the peer
 * to send a reply.  (Does not consume the reply.)  If successful, returns 1
 * and sets #megabitsPerSecond# to the unidirectional bandwidth; else returns 0.
 */
static int
BareTcpBandwidthTo(Socket sock,
                   size_t messageCount,
                   size_t bytesPerMessage,
                   const char *terminator,
                   double *megabitsPerSecond) {

  char *message;
  long elapsedTime;
  int i;
  long startTime;

  message = (char *)malloc(bytesPerMessage);
  memset(message, 0, bytesPerMessage); /* Make purify happy. */
  startTime = MicroTime();

  for(i = 0; i < messageCount && Send(sock, message, bytesPerMessage); i++)
    ; /* Nothing more to do. */

  if(i != messageCount ||
     (terminator != NULL && !Send(sock, terminator, strlen(terminator))) ||
     !CheckIfReadable(sock, 2.0)) {
    free(message);
    return 0;
  }

  elapsedTime = MicroTime() - startTime;
  if(megabitsPerSecond != NULL)
    *megabitsPerSecond =
      ( ((double)(messageCount * bytesPerMessage) * 8.0) /
        ((double)elapsedTime / 1000000.0) ) / 1000000.0;
  free(message);
  return 1;

}


/*
 * Sends #request# on #sock# and waits for the peer to send a reply.  (Does not
 * consume the reply.)  If successful, returns 1 and sets #milliseconds# to the
 * round-trip latency; else returns 0.
 */
static int
BareTcpLatency(Socket sock,
               const char *request,
               double *milliseconds) {

  long elapsedTime;
  long startTime;

  startTime = MicroTime();
  if(!Send(sock, request, strlen(request)) || !CheckIfReadable(sock, 10.0))
    return 0;
  elapsedTime = MicroTime() - startTime;
  if(milliseconds != NULL)
    *milliseconds = (double)elapsedTime / 1000.0;
  return 1;

}


/*
 * Uses the HTTP protocol to measure the unidirectional bandwidth to the peer
 * on #sock# by sending #messageCount# messages, each #bytesPerMessage# long.
 * If successful, returns 1 and sets #megabitsPerSecond# to the bandwidth; else
 * returns 0.
 */
static int
HttpBandwidthTo(Socket sock,
                size_t messageCount,
                size_t bytesPerMessage,
                double *megabitsPerSecond) {
  return 0; /* TBD */
}


/*
 * Uses the HTTP protocol to measure the round-trip latency to the peer on
 * #sock#.  If successful, returns 1 and sets #milliseconds# to the latency;
 * else returns 0.
 */
static int
HttpLatency(Socket sock,
            double *milliseconds) {
  char message[256];
  sprintf(message, HTTP_TRACE HTTP_HOST, AddressMachine(PeerAddress(sock)));
  if(!Send(sock, message, strlen(message)) ||
     !BareTcpLatency(sock, CRLF, milliseconds))
    return 0;
  while(CheckIfReadable(sock, 1.0))
    ReceiveTerminated(sock, message, sizeof(message), LF); /* Toss reply. */
  return 1;
}


/*
 * Uses the NWS sensor protocol to measure the round-trip latency and
 * unidirectional bandwidth (which are combined in the protocol) to the peer on
 * #sock# by sending #messageCount# messages, each #bytesPerMessage# long.  If
 * successful, returns 1 and sets #megabitsPerSecond# to the bandwidth and
 * #milliseconds# to the latency; else returns 0.
 */
static int
NwsSensorBandwidthToAndLatency(Socket sock,
                               size_t messageCount,
                               size_t bytesPerMessage,
                               double *megabitsPerSecond,
                               double *milliseconds) {

  size_t headerSize;
  void *message;
  size_t messageSize;
  struct {
    unsigned int version;
    unsigned int message;
    unsigned int dataSize;
  } nwsHeader;
  struct {
    unsigned int ipAddress;
    unsigned short port;
  } nwsBandwidthTestHandshake;
  struct {
    unsigned int experimentSize;
    unsigned int bufferSize;
    unsigned int messageSize;
  } nwsBandwidthTestRequest;
  int result;
  Socket sockToUse;

  /* Set up the request and translate it into network (XDR) data format. */
  nwsHeader.version  = NWS_VERSION;
  nwsHeader.message  = NWS_TCP_BW_REQ;
  nwsHeader.dataSize = HomogenousNetworkDataSize(UNSIGNED_INT_TYPE, 3);
  nwsBandwidthTestRequest.experimentSize = bytesPerMessage * messageCount;
  nwsBandwidthTestRequest.bufferSize     = SocketSendBufferSize(sock);
  nwsBandwidthTestRequest.messageSize    = bytesPerMessage;

  headerSize = HomogenousNetworkDataSize(UNSIGNED_INT_TYPE, 3);
  messageSize = HomogenousNetworkDataSize(UNSIGNED_INT_TYPE, 3);
  message = malloc(headerSize + messageSize);
  HomogenousConvertHostToNetwork(message, &nwsHeader, UNSIGNED_INT_TYPE, 3);
  HomogenousConvertHostToNetwork
    ((char*)message+headerSize, &nwsBandwidthTestRequest, UNSIGNED_INT_TYPE, 3);

  /* Send the request and receive the handshake with the port to contact. */
  if(!Send(sock, message, headerSize + messageSize) ||
     !Receive(sock, message, headerSize)) {
    free(message);
    return 0;
  }
  HomogenousConvertNetworkToHost(&nwsHeader, message, UNSIGNED_INT_TYPE, 3);
  if(nwsHeader.message != NWS_TCP_HANDSHAKE ||
     !Receive(sock, message, 6)) {
    free(message);
    return 0;
  }
  HomogenousConvertNetworkToHost
    (&nwsBandwidthTestHandshake.ipAddress, message, UNSIGNED_INT_TYPE, 1);
  HomogenousConvertNetworkToHost
    (&nwsBandwidthTestHandshake.port, (char*)message+4, UNSIGNED_SHORT_TYPE, 1);

  /* Open a new connection (if required) and do the actual tests. */
  if(nwsBandwidthTestHandshake.port == NWS_TCP_REUSE)
    sockToUse = sock;
  else if((sockToUse =
    ConnectToPortBuffered(PeerAddress(sock),
                          nwsBandwidthTestHandshake.port,
                          DEFAULT_BUFFER_SIZE,
                          nwsBandwidthTestRequest.bufferSize)) == NO_SOCKET) {
    free(message);
    return 0;
  }

  result =
    BareTcpLatency(sockToUse, " ", milliseconds) &&
    Receive(sockToUse, message, 1) == 1 &&
    BareTcpBandwidthTo
      (sockToUse, messageCount, bytesPerMessage, NULL, megabitsPerSecond);

  if(sockToUse != sock)
    Disconnect(&sockToUse);
  free(message);
  return result;

}


/*
 * Uses the SMTP protocol to measure the unidirectional bandwidth to the peer
 * on #sock# by sending #messageCount# messages, each #bytesPerMessage# long.
 * If successful, returns 1 and sets #megabitsPerSecond# to the bandwidth; else
 * returns 0.
 */
static int
SmtpBandwidthTo(Socket sock,
                size_t messageCount,
                size_t bytesPerMessage,
                double *megabitsPerSecond) {

  const char *myMachine = MyMachineName();
  char message[256];
  int result;

  sprintf(message, SMTP_HELLO, myMachine);
  if(!Send(sock, message, strlen(message)) ||
     !ReceiveTerminated(sock, message, sizeof(message), LF) ||
     strncmp(message, SMTP_OK, strlen(SMTP_OK)) != 0)
    return 0;

  sprintf(message, SMTP_SENDER, myMachine);
  if(!Send(sock, message, strlen(message)) ||
     !ReceiveTerminated(sock, message, sizeof(message), LF) ||
     strncmp(message, SMTP_OK, strlen(SMTP_OK)) != 0)
    return 0;

  sprintf(message, SMTP_RECEIVER, AddressMachine(PeerAddress(sock)));
  if(!Send(sock, message, strlen(message)) ||
     !ReceiveTerminated(sock, message, sizeof(message), LF) ||
     strncmp(message, SMTP_OK, strlen(SMTP_OK)) != 0)
    return 0;

  if(!Send(sock, SMTP_BODY_BEGIN, strlen(message)))
    return 0;
  ReceiveTerminated(sock, message, sizeof(message), LF); /* Toss "Enter" msg. */

  result = BareTcpBandwidthTo
    (sock, messageCount, bytesPerMessage, SMTP_BODY_END, megabitsPerSecond);
  ReceiveTerminated(sock, message, sizeof(message), LF); /* Toss reply. */
  return result;

}


/*
 * Uses the SMTP protocol to measure the round-trip latency to the peer on
 * #sock#.  If successful, returns 1 and sets #milliseconds# to the latency;
 * else returns 0.
 */
static int
SmtpLatency(Socket sock,
            double *milliseconds) {
  char reply[256];
  if(!BareTcpLatency(sock, SMTP_NOOP, milliseconds))
    return 0;
  ReceiveTerminated(sock, reply, sizeof(reply), LF); /* Toss reply. */
  return 1;
}


int
BandwidthTo(Socket sock,
            ApplicationProtocols protocol,
            size_t messageCount,
            size_t bytesPerMessage,
            double *megabitsPerSecond) {

  switch (protocol) {

  case BARE_TCP_PROTOCOL:
    return BareTcpBandwidthTo
      (sock, messageCount, bytesPerMessage, NULL, megabitsPerSecond);

  case HTTP_PROTOCOL:
    return HttpBandwidthTo
      (sock, messageCount, bytesPerMessage, megabitsPerSecond);

  case NWS_SENSOR_PROTOCOL:
    return NwsSensorBandwidthToAndLatency
      (sock, messageCount, bytesPerMessage, megabitsPerSecond, NULL);

  case SMTP_PROTOCOL:
    return SmtpBandwidthTo
      (sock, messageCount, bytesPerMessage, megabitsPerSecond);

  default:
    return 0;

  }

}


int
RoundTripLatency(Socket sock,
                 ApplicationProtocols protocol,
                 double *milliseconds) {

  switch (protocol) {

  case BARE_TCP_PROTOCOL:
    return BareTcpLatency(sock, " ", milliseconds);

  case HTTP_PROTOCOL:
    return HttpLatency(sock, milliseconds);

  case NWS_SENSOR_PROTOCOL:
    return NwsSensorBandwidthToAndLatency(sock, 0, 0, NULL, milliseconds);

  case SMTP_PROTOCOL:
    return SmtpLatency(sock, milliseconds);

  default:
    return 0;

  }

}
