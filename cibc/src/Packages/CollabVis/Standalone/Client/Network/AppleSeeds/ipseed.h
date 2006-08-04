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


#ifndef IPSEED_H
#define IPSEED_H

/*
 * This package provides facilities for establishing IP connections between
 * hosts and sending messages over the connections.  It hides the gory details
 * and configuration problems of the socket.h functions and provides a
 * convenient interface to DNS name/address translation.
 */


#include <sys/types.h> /* size_t */

#ifdef __cplusplus
extern "C" {
#endif


#ifndef NULL
#  define NULL 0
#endif

/* IPv4 address. */
typedef unsigned int ASIP_Address;
/* IPv4 port number. */
typedef unsigned short ASIP_Port;
/* Supported protocols. */
typedef enum {ASIP_TCP_PROTOCOL, ASIP_UDP_PROTOCOL} ASIP_Protocols;
/* A connection to a peer. */
typedef int ASIP_Socket;

/* ASIP_OpenIpPortBuffered wildcard address value. */
#define ASIP_ANY_ADDRESS ((ASIP_Address)0)
/* ASIP_OpenIpPortBuffered wildcard port value. */
#define ASIP_ANY_PORT ((ASIP_Port)0)
/* ASIP_ConnectToIpPortBuffered/ASIP_OpenIpPortBuffered "don't care" value. */
#define ASIP_DEFAULT_BUFFER_SIZE ((size_t)0)
/* Maximum text length of an IP address, i.e., strlen("255.255.255.255"). */
#define ASIP_MAX_IMAGE 15
/* Invalid socket value. */
#define ASIP_NO_SOCKET ((ASIP_Socket)-1)
/* Value for ASIP_CheckIfAnyReadable for blocking wait. */
#define ASIP_WAIT_FOREVER (-1.0)


/*
 * Tries to allow a client to connect to the port associated with #sock#, which
 * must have been returned from ASIP_OpenTcPortBuffered.  Returns a connection
 * to the client if successful, else ASIP_NO_SOCKET.
 */
ASIP_Socket
ASIP_Accept(ASIP_Socket sock);


/*
 * Converts #addr# into a printable string and returns the result.  The value
 * returned is volatile and will be overwritten by subsequent calls.
 */
const char *
ASIP_AddressImage(ASIP_Address addr);


/*
 * Converts #addr# to a fully-qualified machine name and returns the result, or
 * "" on error.  The value returned is volatile and will be overwritten by
 * subsequent calls.
 */
const char *
ASIP_AddressMachine(ASIP_Address addr);


/*
 * Converts #machineOrAddress#, which may be either a DNS name or an IP address
 * image, into a list of addresses.  Copies the list into the #atMost#-long
 * array #addressList#.  Returns the number of addresses copied, or zero on
 * error.  #atMost# may be zero, in which case the function simply returns 1 or
 * 0 depending on whether or not #machineOrAddress# is a valid machine name or
 * IP address image.
 */
int
ASIP_AddressValues(const char *machineOrAddress,
                   ASIP_Address *addressList,
                   unsigned int atMost);
/*
 * Convenience macro for converting a DNS name or an IP address image to a
 * single IP address.
 */
#define ASIP_AddressValue(machineOrAddress,address) \
        ASIP_AddressValues(machineOrAddress,address,1)
/*
 * Convenience macro for determining whether a string represents a valid DNS
 * name or IP address image.
 */
#define ASIP_IsValid(machineOrAddress) \
        ASIP_AddressValues(machineOrAddress,NULL,0)


/*
 * Returns the number of sockets from the ASIP_NO_SOCKET-terminated array
 * #socks# that are readable and copies the readable sockets into the array
 * #readable# (which must be long enough to contain all the elements of
 * #socks#), if it is not NULL.  Waits up to #secondsToWait# seconds for one of
 * the sockets to become readable.
 */
int
ASIP_CheckIfAnyReadable(ASIP_Socket *socks,
                        ASIP_Socket *readable,
                        double secondsToWait);


/* Like ASIP_CheckIfAnyReadable, but for a single socket. */
int
ASIP_CheckIfReadable(ASIP_Socket sock,
                     double secondsToWait);


/*
 * Tries to establish a connection to the #protocol# peer #address#:#port#.
 * For UDP this entails only recording the peer to whom messages on #sock#
 * should be sent; for TCP a full connection is established.  If successful,
 * returns a socket with buffer sizes configured to #receiveBufferSize# and
 * #sendBufferSize#; else returns ASIP_NO_SOCKET.
 */
ASIP_Socket
ASIP_ConnectToIpPortBuffered(ASIP_Protocols protocol,
                             ASIP_Address address,
                             ASIP_Port port,
                             size_t receiveBufferSize,
                             size_t sendBufferSize);
/* Convenience macro for making a connection with default buffer sizes. */
#define ASIP_ConnectToIpPort(protocol,address,port) \
        ASIP_ConnectToIpPortBuffered(protocol, \
                                     address, \
                                     port, \
                                     ASIP_DEFAULT_BUFFER_SIZE, \
                                     ASIP_DEFAULT_BUFFER_SIZE)
/* Convenience macro for making a TCP connection. */
#define ASIP_ConnectToTcpPortBuffered(address,port,receiveSize,sendSize) \
        ASIP_ConnectToIpPortBuffered(ASIP_TCP_PROTOCOL, \
                                     address, \
                                     port, \
                                     receiveSize, \
                                     sendSize)
/* Convenience macro for making a UDP connection. */
#define ASIP_ConnectToUdpPortBuffered(address,port,receiveSize,sendSize) \
        ASIP_ConnectToIpPortBuffered(ASIP_UDP_PROTOCOL, \
                                     address, \
                                     port, \
                                     receiveSize, \
                                     sendSize)
/* Convenience macro for making a TCP connection with default buffer sizes. */
#define ASIP_ConnectToTcpPort(address,port) \
        ASIP_ConnectToIpPort(ASIP_TCP_PROTOCOL,address,port)
/* Convenience macro for making a UDP connection with default buffer sizes. */
#define ASIP_ConnectToUdpPort(address,port) \
        ASIP_ConnectToIpPort(ASIP_UDP_PROTOCOL,address,port)
/* Shorthands for making TCP connections. */
#define ASIP_ConnectToPortBuffered ASIP_ConnectToTcpPortBuffered
#define ASIP_ConnectToPort ASIP_ConnectToTcpPort


/* Disconnects #sock# and sets it to ASIP_NO_SOCKET. */
void
ASIP_Disconnect(ASIP_Socket *sock);


/*
 * Returns the fully-qualified name of the host machine, or "" if the name
 * cannot be determined.
 */
const char *
ASIP_MyMachineName(void);


/*
 * Opens #address#:#port# as a #protocol# listening port.  If successful,
 * returns a socket with buffer sizes configured to #receiveBufferSize# and
 * #sendBufferSize#; else returns ASIP_NO_SOCKET.  If #openedPort# is not NULL,
 * copies the opened port to #openedPort# (useful if #port# is ASIP_ANY_PORT).
 */
ASIP_Socket
ASIP_OpenIpPortBuffered(ASIP_Protocols protocol,
                        ASIP_Address address,
                        ASIP_Port port,
                        size_t receiveBufferSize,
                        size_t sendBufferSize,
                        ASIP_Port *openedPort);
/* Convenience macro for opening a port with default buffer sizes. */
#define ASIP_OpenIpPort(protocol,address,port) \
        ASIP_OpenIpPortBuffered(protocol, \
                                address, \
                                port, \
                                ASIP_DEFAULT_BUFFER_SIZE, \
                                ASIP_DEFAULT_BUFFER_SIZE, \
                                NULL)
/* Convenience macro for opening a TCP port. */
#define ASIP_OpenTcpPortBuffered(address,port,receiveSize,sendSize) \
        ASIP_OpenIpPortBuffered(ASIP_TCP_PROTOCOL, \
                                address, \
                                port, \
                                receiveSize, \
                                sendSize, \
                                NULL)
/* Convenience macro for opening a UDP port. */
#define ASIP_OpenUdpPortBuffered(address,port,receiveSize,sendSize) \
        ASIP_OpenIpPortBuffered(ASIP_UDP_PROTOCOL, \
                                address, \
                                port, \
                                receiveSize, \
                                sendSize, \
                                NULL)
/* Convenience macro for opening a TCP port with default buffer sizes. */
#define ASIP_OpenTcpPort(address,port) \
        ASIP_OpenIpPort(ASIP_TCP_PROTOCOL,address,port)
/* Convenience macro for opening a UDP port with default buffer sizes. */
#define ASIP_OpenUdpPort(address,port) \
        ASIP_OpenIpPort(ASIP_UDP_PROTOCOL,address,port)
/* Shorthands for opening TCP ports. */
#define ASIP_OpenPortBuffered ASIP_OpenTcpPortBuffered
#define ASIP_OpenPort ASIP_OpenTcpPort


/*
 * Returns the address of the peer connected to #sock#, ASIP_ANY_ADDRESS on
 * error.
 */
ASIP_Address
ASIP_PeerAddress(ASIP_Socket sock);


/* Returns the port of the peer connected to #sock#, ASIP_ANY_PORT on error. */
ASIP_Port
ASIP_PeerPort(ASIP_Socket sock);


/*
 * Waits until #size# bytes are received on #sock#, then copies them into
 * #intoWhere#.  If #terminator# is non-null, it points to a caller-defined
 * single-character message terminator, and the function will return as soon as
 * this character is received and copied into #intoWhere#.  Returns the number
 * of characters received.  If #addr# and #port# are not NULL, copies the
 * address and port of the sender into these parameters.  (This is useful only
 * on unconnected UDP sockets, since the sender is otherwise fixed.)
 */
int
ASIP_ReceiveFromTerminated(ASIP_Socket sock,
                           void *intoWhere,
                           size_t size,
                           const char *terminator,
                           ASIP_Address *addr,
                           ASIP_Port *port);
/* Convenience macro for receiving a fixed byte count. */
#define ASIP_ReceiveFrom(sock,intoWhere,size,addr,port) \
  ASIP_ReceiveFromTerminated(sock, intoWhere, size, NULL, addr, port)
/* Convenience macro for receiving with a message terminator. */
#define ASIP_ReceiveTerminated(sock,intoWhere,size,terminators) \
  ASIP_ReceiveFromTerminated(sock, intoWhere, size, terminators, NULL, NULL)
/* Convenience macro for receiving a fixed byte count on a connected socket. */
#define ASIP_Receive(sock,intoWhere,size) \
  ASIP_ReceiveFromTerminated(sock, intoWhere, size, NULL, NULL, NULL)


/*
 * Sends #size# bytes from #fromWhere# as a message on #sock#.  Returns 1 if
 * successful, else 0.  If #addr# is not ASIP_ANY_ADDRESS, the message is sent
 * to #addr#:#port#; otherwise, the message is sent to the peer to which #sock#
 * was previously connected.
 */
int
ASIP_SendTo(ASIP_Socket sock,
            const void *fromWhere,
            size_t size,
            ASIP_Address addr,
            ASIP_Port port);
/* Convenience function for sending on a connected socket. */
#define ASIP_Send(sock,fromWhere,size) ASIP_SendTo(sock, fromWhere, size, ASIP_ANY_ADDRESS, ASIP_ANY_PORT)


/* Returns the local address connected to #sock#, ASIP_ANY_ADDRESS on error. */
ASIP_Address
ASIP_SocketAddress(ASIP_Socket sock);


/*
 * Returns the size, in bytes, of one of #sock#'s buffers.  Returns the send
 * buffer size if #sendSize# is non-zero; otherwise, the receive buffer size.
 */
size_t
ASIP_SocketBufferSize(ASIP_Socket sock,
                      int sendSize);
/* Convenience function for determining the size of a receive buffer. */
#define ASIP_SocketReceiveBufferSize(sock) ASIP_SocketBufferSize(sock, 0)
/* Convenience function for determining the size of a send buffer. */
#define ASIP_SocketSendBufferSize(sock) ASIP_SocketBufferSize(sock, 1)

/* Returns the local port connected to #sock#, ASIP_ANY_PORT on error. */
ASIP_Port
ASIP_SocketPort(ASIP_Socket sock);


#ifdef ASIP_SHORT_NAMES

#define Address ASIP_Address
#define Port ASIP_Port
#define TCP_PROTOCOL ASIP_TCP_PROTOCOL
#define UDP_PROTOCOL ASIP_UDP_PROTOCOL
#define Protocols ASIP_Protocols
#define Socket ASIP_Socket
#define ANY_ADDRESS ASIP_ANY_ADDRESS
#define ANY_PORT ASIP_ANY_PORT
#define DEFAULT_BUFFER_SIZE ASIP_DEFAULT_BUFFER_SIZE
#define MAX_IMAGE ASIP_MAX_IMAGE
#define NO_SOCKET ASIP_NO_SOCKET
#define WAIT_FOREVER ASIP_WAIT_FOREVER

#define Accept ASIP_Accept
#define AddressImage ASIP_AddressImage
#define AddressMachine ASIP_AddressMachine
#define AddressValues ASIP_AddressValues
#define AddressValue ASIP_AddressValue
#define IsValid ASIP_IsValid
#define CheckIfAnyReadable ASIP_CheckIfAnyReadable
#define CheckIfReadable ASIP_CheckIfReadable
#define ConnectToIpPortBuffered ASIP_ConnectToIpPortBuffered
#define ConnectToIpPort ASIP_ConnectToIpPort
#define ConnectToTcpPortBuffered ASIP_ConnectToTcpPortBuffered
#define ConnectToUdpPortBuffered ASIP_ConnectToUdpPortBuffered
#define ConnectToTcpPort ASIP_ConnectToTcpPort
#define ConnectToUdpPort ASIP_ConnectToUdpPort
#define ConnectToPortBuffered ASIP_ConnectToPortBuffered
#define ConnectToPort ASIP_ConnectToPort
#define Disconnect ASIP_Disconnect
#define MyMachineName ASIP_MyMachineName
#define OpenIpPortBuffered ASIP_OpenIpPortBuffered
#define OpenIpPort ASIP_OpenIpPort
#define OpenTcpPortBuffered ASIP_OpenTcpPortBuffered
#define OpenUdpPortBuffered ASIP_OpenUdpPortBuffered
#define OpenTcpPort ASIP_OpenTcpPort
#define OpenUdpPort ASIP_OpenUdpPort
#define OpenPortBuffered ASIP_OpenPortBuffered
#define OpenPort ASIP_OpenPort
#define PeerAddress ASIP_PeerAddress
#define PeerPort ASIP_PeerPort
#define ReceiveFromTerminated ASIP_ReceiveFromTerminated
#define ReceiveTerminated ASIP_ReceiveTerminated
#define ReceiveFrom ASIP_ReceiveFrom
// conflicts with RMF Receive
//#define Receive ASIP_Receive
#define SendTo ASIP_SendTo
// conflicts with RMF Send
//#define Send ASIP_Send
#define SocketAddress ASIP_SocketAddress
#define SocketBufferSize ASIP_SocketBufferSize
#define SocketReceiveBufferSize ASIP_SocketReceiveBufferSize
#define SocketSendBufferSize ASIP_SocketSendBufferSize
#define SocketPort ASIP_SocketPort

#endif


#ifdef __cplusplus
}
#endif

#endif
