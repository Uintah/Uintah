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


#ifndef NETRESOURCESEED_H
#define NETRESOURCESEED_H


/*
 * This package provides facilities for measuring network resource usage and
 * availability.  It is presently still under development.
 */


#ifdef __cplusplus
extern "C" {
#endif


#include "ipseed.h"


/* Protocols supported by the package. */
typedef enum {
  ASNETRES_BARE_TCP_PROTOCOL, ASNETRES_HTTP_PROTOCOL,
  ASNETRES_NWS_SENSOR_PROTOCOL, ASNETRES_SMTP_PROTOCOL
} ASNETRES_ApplicationProtocols;


/*
 * Uses #protocol# to receive #messageCount# messages on #sock#, each
 * #bytesPerMessage# bytes long, to measure the unidirectional bandwidth.  If
 * successful, returns 1 and sets #megabitsPerSecond# to the bandwidth; else
 * returns 0.
 */
int
ASNETRES_BandwidthFrom(Socket sock,
                       ASNETRES_ApplicationProtocols protocol,
                       size_t messageCount,
                       size_t bytesPerMessage,
                       double *megabitsPerSecond);


/*
 * Uses #protocol# to send #messageCount# messages on #sock#, each
 * #bytesPerMessage# bytes long, to measure the unidirectional bandwidth.  If
 * successful, returns 1 and sets #megabitsPerSecond# to the bandwidth; else
 * returns 0.
 */
int
ASNETRES_BandwidthTo(Socket sock,
                     ASNETRES_ApplicationProtocols protocol,
                     size_t messageCount,
                     size_t bytesPerMessage,
                     double *megabitsPerSecond);


/*
 * Uses #protocol# to send a message on #sock# to measure the round-trip
 * latency.  If successful, returns 1 and sets #milliseconds# to the latency;
 * else returns 0.
 */
int
ASNETRES_RoundTripLatency(Socket sock,
                          ASNETRES_ApplicationProtocols protocol,
                          double *milliseconds);


#ifdef ASNETRES_SHORT_NAMES

#define BARE_TCP_PROTOCOL ASNETRES_BARE_TCP_PROTOCOL
#define HTTP_PROTOCOL ASNETRES_HTTP_PROTOCOL
#define NWS_SENSOR_PROTOCOL ASNETRES_NWS_SENSOR_PROTOCOL
#define SMTP_PROTOCOL ASNETRES_SMTP_PROTOCOL
#define ApplicationProtocols ASNETRES_ApplicationProtocols

#define BandwidthFrom ASNETRES_BandwidthFrom
#define BandwidthTo ASNETRES_BandwidthTo
#define RoundTripLatency ASNETRES_RoundTripLatency

#endif


#ifdef __cplusplus  
}  
#endif


#endif
