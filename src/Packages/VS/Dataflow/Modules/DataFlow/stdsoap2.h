/*

stdsoap2.h 2.6.2

Runtime environment.

gSOAP XML Web services tools
Copyright (C) 2000-2004, Robert van Engelen, Genivia, Inc. All Rights Reserved.

Contributors:

Wind River Systems, Inc., for the following additions (marked WR[...]) :
  - vxWorks compatible
  - Support for IPv6.

--------------------------------------------------------------------------------
gSOAP public license.

The contents of this file are subject to the gSOAP Public License Version 1.3
(the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at
http://www.cs.fsu.edu/~engelen/soaplicense.html
Software distributed under the License is distributed on an "AS IS" basis,
WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
for the specific language governing rights and limitations under the License.

The Initial Developer of the Original Code is Robert A. van Engelen.
Copyright (C) 2000-2004 Robert A. van Engelen, Genivia inc. All Rights Reserved.
--------------------------------------------------------------------------------
GPL license.

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program; if not, write to the Free Software Foundation, Inc., 59 Temple
Place, Suite 330, Boston, MA 02111-1307 USA

Author contact information:
engelen@genivia.com / engelen@acm.org
--------------------------------------------------------------------------------
*/

#ifdef WITH_SOAPDEFS_H
# include "soapdefs.h"		/* include user-defined stuff */
#endif

#ifndef _THREAD_SAFE
# define _THREAD_SAFE
#endif

#ifndef OPENSERVER
# ifndef _REENTRANT
#  define _REENTRANT
# endif
#endif

#ifndef SOAP_BEGIN_NAMESPACE
# define SOAP_BEGIN_NAMESPACE(name)
#endif

#ifndef SOAP_END_NAMESPACE
# define SOAP_END_NAMESPACE(name)
#endif

#ifndef SOAP_FMAC1	/* stdsoap2.h declaration macro */
# define SOAP_FMAC1
#endif

#ifndef SOAP_FMAC2	/* stdsoap2.h declaration macro */
# define SOAP_FMAC2
#endif

#ifndef SOAP_FMAC3	/* (de)serializer declaration macro */
# define SOAP_FMAC3
#endif

#ifndef SOAP_FMAC4	/* (de)serializer declaration macro */
# define SOAP_FMAC4
#endif

#ifndef SOAP_FMAC5	/* stub/skeleton declaration macro */
# define SOAP_FMAC5
#endif

#ifndef SOAP_FMAC6	/* stub/skeleton declaration macro */
# define SOAP_FMAC6
#endif

#ifndef SOAP_CMAC	/* class declaration macro */
# define SOAP_CMAC
#endif

#ifndef SOAP_NMAC	/* namespace table declaration macro */
# define SOAP_NMAC
#endif

#ifndef SOAP_SOURCE_STAMP
# define SOAP_SOURCE_STAMP(str)
#endif

#ifdef WITH_LEANER
# ifndef WITH_LEAN
#  define WITH_LEAN
# endif
#endif

#ifdef WITH_LEAN
# ifdef WITH_COOKIES
#  error "Cannot build WITH_LEAN code WITH_COOKIES enabled"
# endif
#endif

#ifndef STDSOAP_H
#define STDSOAP_H

/* WR[ */
#if (defined(__vxworks) || defined(__VXWORKS__))
# define VXWORKS
#endif
/* ]WR */

#ifdef _WIN32
# ifndef WIN32
#  define WIN32
# endif
#endif

#ifdef UNDER_CE
# ifndef WIN32
#  define WIN32
# endif
#endif

#ifdef __BORLANDC__
# ifdef __WIN32__
#  ifndef WIN32
#   define WIN32
#  endif
# endif
#endif

#ifdef __CYGWIN__
# ifndef CYGWIN
#  define CYGWIN
# endif
#endif

#ifdef __SYMBIAN32__ 
# define SYMBIAN
#endif

#ifdef __palmos__
# define PALM
#endif

#ifdef PALM_GCC
# define PALM
#endif

#ifdef HAVE_CONFIG_H
# include "config.h"
#else
# if defined(UNDER_CE)
#  define WITH_LEAN
# elif defined(WIN32)
#  define HAVE_STRRCHR
#  define HAVE_STRTOD
#  define HAVE_STRTOL
#  define HAVE_STRTOUL
#  define HAVE_SYS_TIMEB_H
#  define HAVE_FTIME
#  define HAVE_WCTOMB
#  define HAVE_MBTOWC
# elif defined(CYGWIN)
#  define HAVE_STRRCHR
#  define HAVE_STRTOD
#  define HAVE_STRTOL
#  define HAVE_STRTOUL
#  define HAVE_SYS_TIMEB_H
#  define HAVE_FTIME
#  define HAVE_RAND_R
#  define HAVE_GMTIME_R
#  define HAVE_LOCALTIME_R
#  define HAVE_WCTOMB
#  define HAVE_MBTOWC
# elif defined(__APPLE__)
#  define HAVE_STRRCHR
#  define HAVE_STRTOD
#  define HAVE_STRTOL
#  define HAVE_STRTOUL
#  define HAVE_RAND_R
#  define HAVE_GMTIME_R
#  define HAVE_LOCALTIME_R
#  define HAVE_TIMEGM
#  define HAVE_WCTOMB
#  define HAVE_MBTOWC
# elif defined(_AIXVERSION_431)
#  define HAVE_STRRCHR
#  define HAVE_STRTOD
#  define HAVE_STRTOL
#  define HAVE_STRTOUL
#  define HAVE_SYS_TIMEB_H
#  define HAVE_FTIME
#  define HAVE_RAND_R
#  define HAVE_GMTIME_R
#  define HAVE_LOCALTIME_R
#  define HAVE_WCTOMB
#  define HAVE_MBTOWC
# elif defined(HP_UX)
#  define HAVE_STRRCHR
#  define HAVE_STRTOD
#  define HAVE_STRTOL
#  define HAVE_STRTOUL
#  define HAVE_SYS_TIMEB_H
#  define HAVE_FTIME
#  define HAVE_RAND_R
#  define HAVE_GMTIME_R
#  define HAVE_LOCALTIME_R
#  define HAVE_WCTOMB
#  define HAVE_MBTOWC
# elif defined(FREEBSD)
#  define HAVE_STRRCHR
#  define HAVE_STRTOD
#  define HAVE_STRTOL
#  define HAVE_STRTOUL
#  define HAVE_GETTIMEOFDAY
#  define HAVE_RAND_R
#  define HAVE_GMTIME_R
#  define HAVE_LOCALTIME_R
#  define HAVE_WCTOMB
#  define HAVE_MBTOWC
# elif defined(__VMS)
#  define HAVE_STRRCHR
#  define HAVE_STRTOD
#  define HAVE_STRTOL
#  define HAVE_STRTOUL
#  define HAVE_SYS_TIMEB_H
#  define HAVE_FTIME
#  define HAVE_RAND_R
#  define HAVE_GMTIME_R
#  define HAVE_LOCALTIME_R
#  define HAVE_WCTOMB
#  define HAVE_MBTOWC
# elif defined(__GLIBC__)
#  define HAVE_STRRCHR
#  define HAVE_STRTOD
#  define HAVE_STRTOL
#  define HAVE_STRTOUL
#  define HAVE_SYS_TIMEB_H
#  define HAVE_FTIME
#  define HAVE_RAND_R
#  define HAVE_GMTIME_R
#  define HAVE_LOCALTIME_R
#  define HAVE_TIMEGM
#  define HAVE_WCTOMB
#  define HAVE_MBTOWC
# elif defined(TRU64)
#  define HAVE_STRRCHR
#  define HAVE_STRTOD
#  define HAVE_STRTOL
#  define HAVE_STRTOUL
#  define HAVE_GETTIMEOFDAY
#  define HAVE_SYS_TIMEB_H
#  define HAVE_RAND_R
#  define HAVE_GMTIME_R
#  define HAVE_LOCALTIME_R
#  define HAVE_WCTOMB
#  define HAVE_MBTOWC
# elif defined(MAC_CARBON)
#  define HAVE_STRRCHR
#  define HAVE_STRTOD
#  define HAVE_STRTOL
#  define HAVE_STRTOUL
#  define HAVE_FTIME
#  define HAVE_RAND_R
#  define HAVE_GETHOSTBYNAME_R
#  define HAVE_GMTIME_R
#  define HAVE_LOCALTIME_R
#  define HAVE_WCTOMB
#  define HAVE_MBTOWC
# elif defined(PALM)
#  define HAVE_STRTOD	/* strtod() is defined in palmmissing.h */
#  ifndef CONST2
#   define CONST2
#  endif
#  define WITH_LEAN
#  define WITH_NONAMESPACES
#  define _LINUX_CTYPE_H
#  include <sys_types.h>
   typedef WChar wchar_t;
#  define IGNORE_STDIO_STUBS
   typedef Int32 time_t;
#  define tm HostTmType
#  define strftime HostStrFTime
#  define mktime HostMkTime
#  define localtime HostLocalTime
#  define tm_year tm_year_
#  define tm_hour tm_hour_
#  define tm_mon tm_mon_
#  define tm_min tm_min_
#  define tm_sec tm_sec_
#  define tm_mday tm_mday_
#  define tm_isdst tm_isdst_
#  include <StdIOPalm.h>
#  define O_NONBLOCK FNONBIO
#  include <sys_time.h>
#  if 1
    void displayText(char *text);
#   define pdebug(s) displayText(s) 
#   define pdebugV(s,p) sprintf(buff,s,p); displayText(buff)
#  else
#   define pdebug(s) WinDrawChars(s,strlen(s),10,10) 
#   define pdebugV(s,p) sprintf(buff,s,p); WinDrawChars(buff,strlen(buff),10,10)
#  endif
#  include "palmmissing.h"
#  include "slib_socket.h"
# elif defined(SYMBIAN)
#  define WITH_LEAN
#  define WITH_NONAMESPACES
#  define CONST2 const
#  undef SOAP_FMAC1
#  define SOAP_FMAC1 EXPORT_C
#  ifdef __cplusplus
#   include <e32std.h>
#  else
#   define TInt64 long
#  endif
/* WR[ */
# elif defined(VXWORKS)
#  define HAVE_STRRCHR
#  define HAVE_STRTOD
#  define HAVE_STRTOL
#  define HAVE_STRTOUL
#  define HAVE_RAND_R
#  define HAVE_PGMTIME_R
#  define HAVE_PLOCALTIME_R
#  define HAVE_MKTIME
/* ]WR */
# else
/* Default asumptions on supported functions */
#  define HAVE_STRRCHR
#  define HAVE_STRTOD
#  define HAVE_STRTOL
#  define HAVE_STRTOUL
#  define HAVE_SYS_TIMEB_H
#  define HAVE_FTIME
#  define HAVE_RAND_R
#  define HAVE_GETHOSTBYNAME_R
#  define HAVE_GMTIME_R
#  define HAVE_LOCALTIME_R
#  define HAVE_WCTOMB
#  define HAVE_MBTOWC
# endif
#endif

#if defined(TRU64)
# define SOAP_LONG_FORMAT "%ld"
# define SOAP_ULONG_FORMAT "%lu"
#elif defined(WIN32)
# define SOAP_LONG_FORMAT "%I64d"
# define SOAP_ULONG_FORMAT "%I64u"
#endif

#ifndef SOAP_LONG_FORMAT
# define SOAP_LONG_FORMAT "%lld"	/* printf format for 64 bit ints */
#endif

#ifndef SOAP_ULONG_FORMAT
# define SOAP_ULONG_FORMAT "%llu"	/* printf format for unsigned 64 bit ints */
#endif

#ifndef SOAP_MALLOC			/* use libc malloc */
# define SOAP_MALLOC(n) malloc(n)
#endif

#ifndef SOAP_FREE			/* use libc free */
# define SOAP_FREE(p) free(p)
#endif

#include <stdlib.h>

#ifndef PALM
# include <stdio.h>
#endif

#ifndef PALM_GCC
# include <string.h>
#endif

#include <ctype.h>	/* for tolower() */
#include <limits.h>

#if defined(__cplusplus) && !defined(UNDER_CE)
# include <string>
# include <iostream>
  using namespace std;
#endif

#ifndef UNDER_CE
# ifndef PALM
#  include <errno.h>
#  ifndef MAC_CARBON
#   include <sys/types.h>
#  endif
#  ifndef WITH_LEAN
#   ifdef HAVE_SYS_TIMEB_H
#    include <sys/timeb.h>		/* for ftime() */
#   endif
#   include <time.h>
#  endif
# endif
#endif

#ifdef OPENSERVER
# include <sys/socket.h>
# include <sys/stream.h>
# include <sys/protosw.h>
  extern int h_errno;
#endif

#ifndef MAC_CARBON
# ifndef WIN32
#  ifndef PALM
#   include <sys/socket.h>
/* WR[ */
#   ifdef VXWORKS
#    include <sockLib.h>
#   endif
#   ifndef VXWORKS
/* ]WR */
#    ifndef SYMBIAN
#     include <strings.h>
#    endif
/* WR[ */
#   endif
/* ]WR */
#   ifdef SUN_OS
#    include <sys/stream.h>		/* SUN */
#    include <sys/socketvar.h>		/* SUN < 2.8 (?) */
#   endif
/* WR[ */
#   ifdef VXWORKS
#    include <sys/times.h>
#   else
/* ]WR */
#    include <sys/time.h>
/* WR[ */
#   endif
/* ]WR */
#   include <netinet/in.h>
#   include <netinet/tcp.h>		/* TCP_NODELAY */
#   include <arpa/inet.h>
#  endif
# endif
#endif

#ifdef WITH_FASTCGI
# include <fcgi_stdio.h>
#endif

#ifdef WITH_OPENSSL
# define OPENSSL_NO_KRB5
# include <openssl/ssl.h>
# include <openssl/err.h>
# include <openssl/rand.h>
# ifndef ALLOW_OLD_VERSIONS
#  if (OPENSSL_VERSION_NUMBER < 0x00905100L)
#   error "Must use OpenSSL 0.9.6 or later"
#  endif
# endif
#endif

#ifdef WITH_GZIP
# ifndef WITH_ZLIB
#  define WITH_ZLIB
# endif
#endif

#ifdef WITH_CASEINSENSITIVETAGS
# define SOAP_STRCMP soap_tag_cmp	/* case insensitve XML element/attribute names */
#else
# define SOAP_STRCMP strcmp		/* case sensitive XML element/attribute names */
#endif

#ifdef WITH_ZLIB
# include <zlib.h>
#endif

#ifndef PALM_GCC
# include <math.h>	/* for isnan() */
#endif

/* #define DEBUG */ /* Uncomment to debug sending (in file SENT.log) receiving (in file RECV.log) and messages (in file TEST.log) */

#ifdef __cplusplus
extern "C" {
#endif

#define soap_get0(soap) (((soap)->bufidx>=(soap)->buflen && soap_recv(soap)) ? EOF : (unsigned char)(soap)->buf[(soap)->bufidx])
#define soap_get1(soap) (((soap)->bufidx>=(soap)->buflen && soap_recv(soap)) ? EOF : (unsigned char)(soap)->buf[(soap)->bufidx++])
#define soap_revget1(soap) ((soap)->bufidx--)
#define soap_unget(soap, c) ((soap)->ahead = c)
#define soap_register_plugin(soap, plugin) soap_register_plugin_arg(soap, plugin, NULL)
#define soap_imode(soap, n) ((soap)->mode = (soap)->imode = (n))
#define soap_set_imode(soap, n) ((soap)->mode = (soap)->imode |= (n))
#define soap_clr_imode(soap, n) ((soap)->mode = (soap)->imode &= ~(n))
#define soap_omode(soap, n) ((soap)->mode = (soap)->omode = (n))
#define soap_set_omode(soap, n) ((soap)->mode = (soap)->omode |= (n))
#define soap_clr_omode(soap, n) ((soap)->mode = (soap)->omode &= ~(n))
#define soap_destroy(soap) soap_delete((soap), NULL)

#ifdef WIN32
# ifndef UNDER_CE
#  include <io.h>
#  include <fcntl.h>
# endif
# include <winsock.h>
/* # include <winsock2.h> */ /* Alternative: use winsock2 (not available with eVC) */
/* WR[ */
# ifdef WITH_IPV6
#  include <ws2tcpip.h>
#  include <wspiapi.h>
# endif
#else
# ifdef VXWORKS
#  include <hostLib.h>
#  include <ioctl.h>
#  include <ioLib.h>
# endif
/* ]WR */
# ifndef MAC_CARBON
#  ifndef PALM
#   include <netdb.h>
#   include <netinet/in.h>
#  endif
#  include <unistd.h>
#  include <fcntl.h>
# endif
#endif

#ifdef WIN32
# define SOAP_SOCKET SOCKET
#else
# define SOAP_SOCKET int
# define closesocket(n) close(n)
#endif

#define soap_valid_socket(n) ((n) >= 0)
#define SOAP_INVALID_SOCKET (-1)

/* WR[ */
#ifdef VXWORKS
# ifdef __INCmathh 
#  define _MATH_H
#   include <private/mathP.h>
#   define isnan(num) isNan(num)
# endif
#endif

#ifdef WIN32 
# define _MATH_H
# include <float.h>
# define isnan(num) _isnan(num)
#endif
/* ]WR */

#if (!defined(_MATH_H) && !defined(_MATH_INCLUDED))
# ifndef isnan
#  define isnan(_) (0)
# endif
#endif

extern const struct soap_double_nan { unsigned int n1, n2; } soap_double_nan;

#if defined(SYMBIAN)
# define LONG64 TInt64
# define ULONG64 TInt64
#elif !defined(WIN32)
# define LONG64 long long
# define ULONG64 unsigned LONG64
#elif defined(UNDER_CE)
# define LONG64 __int64
# define ULONG64 unsigned LONG64
#elif defined(__BORLANDC__)
# define LONG64 __int64
# define ULONG64 unsigned LONG64
#endif

#ifdef WIN32
# define SOAP_EINTR WSAEINTR
# define SOAP_EAGAIN WSAEWOULDBLOCK
# define SOAP_EWOULDBLOCK WSAEWOULDBLOCK
# define SOAP_EINPROGRESS WSAEINPROGRESS
#else
# define SOAP_EINTR EINTR
# define SOAP_EAGAIN EAGAIN
# ifdef SYMBIAN
#  define SOAP_EWOULDBLOCK 9898
#  define SOAP_EINPROGRESS 9899
# else
#  define SOAP_EWOULDBLOCK EWOULDBLOCK
#  define SOAP_EINPROGRESS EINPROGRESS
# endif
#endif

#ifdef WIN32
# ifdef UNDER_CE
#  define soap_errno GetLastError()
#  define soap_socket_errno GetLastError()
# else
#  define soap_errno GetLastError()
#  define soap_socket_errno WSAGetLastError()
# endif
#else
# define soap_errno errno
# define soap_socket_errno errno
#endif

#ifndef SOAP_BUFLEN
# ifndef WITH_LEAN
#  define SOAP_BUFLEN  (32768) /* buffer length for socket packets, also used by gethostbyname_r so don't make this too small */
# else
#  define SOAP_BUFLEN   (2048)
# endif
#endif
#ifndef SOAP_LABLEN
# ifndef WITH_LEAN
#  define SOAP_LABLEN  (256) /* initial look-aside buffer length */
# else
#  define SOAP_LABLEN   (64)
# endif
#endif
#ifndef SOAP_PTRHASH
# ifndef WITH_LEAN
#  define SOAP_PTRHASH  (1024) /* size of pointer analysis hash table (must be power of 2) */
# else
#  define SOAP_PTRHASH    (16)
# endif
#endif
#ifndef SOAP_IDHASH
# ifndef WITH_LEAN
#  define SOAP_IDHASH    (256) /* size of hash table for receiving id/href's */
# else
#  define SOAP_IDHASH     (16)
# endif
#endif
#ifndef SOAP_BLKLEN
# define SOAP_BLKLEN     (256) /* size of blocks to collect long strings and XML attributes */
#endif
#ifndef SOAP_TAGLEN
# define SOAP_TAGLEN     (256) /* maximum length of XML element tag/attribute name + 1 */
#endif
#ifndef SOAP_HDRLEN
# ifndef WITH_LEAN
#  define SOAP_HDRLEN   (8192) /* maximum length of HTTP header line (must be >4096 to read cookies) */
# else
#  define SOAP_HDRLEN   (1024)
# endif
#endif
#ifndef SOAP_MAXDIMS
# define SOAP_MAXDIMS	 (16) /* maximum array dimensions (array nestings) must be less than 64 to protect soap->tmpbuf */
#endif

#ifndef SOAP_MAXLOGS
# define SOAP_MAXLOGS	  (3) /* max number of debug logs per struct soap environment */
# define SOAP_INDEX_RECV  (0)
# define SOAP_INDEX_SENT  (1)
# define SOAP_INDEX_TEST  (2)
#endif

#ifndef SOAP_MAXKEEPALIVE
# define SOAP_MAXKEEPALIVE (100) /* max iterations to keep server connection alive */
#endif

#ifndef SOAP_MAXARRAYSIZE
# define SOAP_MAXARRAYSIZE (100000) /* "trusted" max size of inbound SOAP array for compound array allocation */
#endif

/* WR[ */
#ifdef VXWORKS
# ifndef FLT_MAX
#  define FLT_MAX _ARCH_FLT_MAX
# endif
# ifndef DBL_MAX
#  define DBL_MAX _ARCH_DBL_MAX
# endif
#endif
/* ]WR */

#ifndef FLT_NAN
# if (defined(_MATH_H) || defined(_MATH_INCLUDED))
#  define FLT_NAN (*(float*)&soap_double_nan)
# else
#  define FLT_NAN (0.0)
# endif
#endif

#ifndef FLT_PINFTY
# ifdef FLT_MAX
#  define FLT_PINFTY FLT_MAX
# else
#  ifdef HUGE_VAL
#    define FLT_PINFTY (float)HUGE_VAL
#  else
#   ifdef FLOAT_MAX
#    define FLT_PINFTY FLOAT_MAX
#   else
#    define FLT_PINFTY (3.40282347e+38)
#   endif
#  endif
# endif
#endif

#ifndef FLT_NINFTY
# define FLT_NINFTY (-FLT_PINFTY)
#endif

#ifndef DBL_NAN
# if (defined(_MATH_H) || defined(_MATH_INCLUDED))
#  define DBL_NAN (*(double*)&soap_double_nan)
# else
#  define DBL_NAN (0.0)
# endif
#endif
#ifndef DBL_PINFTY
# ifdef DBL_MAX
#  define DBL_PINFTY DBL_MAX
# else
#  ifdef HUGE_VAL
#   define DBL_PINFTY (double)HUGE_VAL
#  else
#   ifdef DOUBLE_MAX
#    define DBL_PINFTY DOUBLE_MAX
#   else
#    define DBL_PINFTY (1.7976931348623157e+308)
#   endif
#  endif
# endif
#endif

#ifndef DBL_NINFTY
# define DBL_NINFTY (-DBL_PINFTY)
#endif

/* gSOAP error codes */

#define SOAP_EOF			EOF
#define SOAP_ERR			EOF
#define SOAP_OK				0
#define SOAP_CLI_FAULT			1
#define SOAP_SVR_FAULT			2
#define SOAP_TAG_MISMATCH		3
#define SOAP_TYPE			4
#define SOAP_SYNTAX_ERROR		5
#define SOAP_NO_TAG			6
#define SOAP_IOB			7
#define SOAP_MUSTUNDERSTAND		8
#define SOAP_NAMESPACE			9
#define SOAP_OBJ_MISMATCH		10
#define SOAP_FATAL_ERROR		11
#define SOAP_FAULT			12
#define SOAP_NO_METHOD			13
#define SOAP_GET_METHOD			14
#define SOAP_EOM			15
#define SOAP_NULL			16
#define SOAP_MULTI_ID			17
#define SOAP_MISSING_ID			18
#define SOAP_HREF			19
#define SOAP_TCP_ERROR			20
#define SOAP_HTTP_ERROR			21
#define SOAP_SSL_ERROR			22
#define SOAP_ZLIB_ERROR			23
#define SOAP_DIME_ERROR			24
#define SOAP_EOD			25
#define SOAP_VERSIONMISMATCH		26
#define SOAP_DIME_MISMATCH		27
#define SOAP_PLUGIN_ERROR		28
#define SOAP_DATAENCODINGUNKNOWN	29
#define SOAP_REQUIRED			30
#define SOAP_OCCURS			31

#define soap_xml_error_check(e) ((e) == SOAP_TAG_MISMATCH || (e) == SOAP_SYNTAX_ERROR || (e) == SOAP_NAMESPACE || (e) == SOAP_MULTI_ID || (e) == SOAP_MISSING_ID || (e) == SOAP_REQUIRED || (e) == SOAP_OCCURS)
#define soap_soap_error_check(e) ((e) == SOAP_CLI_FAULT || (e) == SOAP_SVR_FAULT || (e) == SOAP_VERSIONMISMATCH || (e) == SOAP_MUSTUNDERSTAND || (e) == SOAP_FAULT || (e) == SOAP_NO_METHOD || (e) == SOAP_OBJ_MISMATCH || (e) == SOAP_NULL)
#define soap_tcp_error_check(e) ((e) == SOAP_EOF || (e) == SOAP_TCP_ERROR)
#define soap_ssl_error_check(e) ((e) == SOAP_SSL_ERROR)
#define soap_zlib_error_check(e) ((e) == SOAP_ZLIB_ERROR)
#define soap_dime_error_check(e) ((e) == SOAP_DIME_ERROR || (e) == SOAP_DIME_MISMATCH)
#define soap_http_error_check(e) ((e) == SOAP_HTTP_ERROR || (e) == SOAP_GET_METHOD || ((e) >= 100 && (e) < 600))

/* gSOAP HTTP response status codes 100 to 600 are reserved */

/* Special gSOAP HTTP response status codes */

#define SOAP_STOP		1000	/* No HTTP response */
#define SOAP_HTML		1001	/* Custom HTML response */
#define SOAP_FILE		1002	/* Custom file-based response */

/* gSOAP HTTP request status codes */

#define SOAP_POST		1003
#define SOAP_GET		1104

/* gSOAP DIME */

#define SOAP_DIME_CF		0x01
#define SOAP_DIME_ME		0x02
#define SOAP_DIME_MB		0x04
#define SOAP_DIME_VERSION	0x08 /* DIME version 1 */
#define SOAP_DIME_MEDIA		0x10
#define SOAP_DIME_ABSURI	0x20

/* gSOAP ZLIB */

#define SOAP_ZLIB_NONE		0x00
#define SOAP_ZLIB_DEFLATE	0x01
#define SOAP_ZLIB_INFLATE	0x02
#define SOAP_ZLIB_GZIP		0x02

/* gSOAP transport, connection, and content encoding modes */

#define SOAP_IO			0x000003
#define SOAP_IO_FLUSH		0x000000	/* flush output immediately, no buffering */
#define SOAP_IO_BUFFER		0x000001	/* buffer output in packets of size SOAP_BUFLEN */
#define SOAP_IO_STORE		0x000002	/* store entire output to determine length for transport */
#define SOAP_IO_CHUNK		0x000003	/* use HTTP chunked transfer AND buffer packets */

#define SOAP_IO_LENGTH		0x000004
#define SOAP_IO_KEEPALIVE	0x000008

#define SOAP_ENC_XML		0x000010	/* plain XML encoding, no HTTP header */
#define SOAP_ENC_DIME		0x000020
#define SOAP_ENC_ZLIB		0x000040
#define SOAP_ENC_SSL		0x000080

#define SOAP_XML_STRICT		0x000100	/* input mode flag */
#define SOAP_XML_CANONICAL	0x000200	/* output mode flag */
#define SOAP_XML_TREE		0x000400
#define SOAP_XML_GRAPH		0x000800
#define SOAP_XML_NIL		0x001000
#define SOAP_XML_DOM		0x002000

#define SOAP_C_NOIOB		0x010000
#define SOAP_C_UTFSTRING	0x020000
#define SOAP_C_MBSTRING		0x040000
#define SOAP_C_LATIN		0x080000

#define SOAP_DOM_TREE		0x100000
#define SOAP_DOM_NODE		0x200000

#define SOAP_IO_DEFAULT		SOAP_IO_FLUSH

/* SSL client/server authentication settings */

#define SOAP_SSL_NO_AUTHENTICATION		0x00	/* for testing purposes */
#define SOAP_SSL_REQUIRE_SERVER_AUTHENTICATION	0x01	/* client requires server to authenticate */
#define SOAP_SSL_REQUIRE_CLIENT_AUTHENTICATION	0x02	/* server requires client to authenticate */

#define SOAP_SSL_DEFAULT			SOAP_SSL_REQUIRE_SERVER_AUTHENTICATION

/* */

#define SOAP_BEGIN		0
#define SOAP_IN_ENVELOPE	2
#define SOAP_IN_HEADER		3
#define SOAP_END_HEADER		4
#define SOAP_IN_BODY		5
#define SOAP_END_BODY		6
#define SOAP_END_ENVELOPE	7
#define SOAP_END		8

/* DEBUG macros */

#ifndef WITH_LEAN
# ifdef DEBUG
#  ifndef SOAP_DEBUG
#   define SOAP_DEBUG
#  endif
# endif
#endif

#ifdef SOAP_DEBUG
# ifndef SOAP_MESSAGE
#  define SOAP_MESSAGE fprintf
# endif
# ifndef DBGLOG
#  define DBGLOG(DBGFILE, CMD) \
{ if (soap)\
  { if (!soap->fdebug[SOAP_INDEX_##DBGFILE])\
      soap_open_logfile(soap, SOAP_INDEX_##DBGFILE);\
    if (soap->fdebug[SOAP_INDEX_##DBGFILE])\
    { FILE *fdebug = soap->fdebug[SOAP_INDEX_##DBGFILE];\
      CMD;\
      fflush(fdebug);\
    }\
  }\
}
# endif
# ifndef DBGMSG
#  define DBGMSG(DBGFILE, MSG, LEN) \
{ if (soap)\
  { if (!soap->fdebug[SOAP_INDEX_##DBGFILE])\
      soap_open_logfile(soap, SOAP_INDEX_##DBGFILE);\
    if (soap->fdebug[SOAP_INDEX_##DBGFILE])\
    { fwrite((MSG), 1, (LEN), soap->fdebug[SOAP_INDEX_##DBGFILE]);\
      fflush(soap->fdebug[SOAP_INDEX_##DBGFILE]);\
    }\
  }\
}
# endif
#else
# define DBGLOG(DBGFILE, CMD)
# define DBGMSG(DBGFILE, MSG, LEN)
#endif

typedef long wchar; /* 32 bit, for compatibility */

struct Namespace
{ const char *id;
  const char *ns;
  const char *in;
  char *out;
};

struct soap_nlist
{ struct soap_nlist *next;
  unsigned int level;
  short index; /* corresponding entry in ns mapping table */
  char *ns; /* only set when parsed ns URI is not in the ns mapping table */
  char id[1]; /* the actual string value overflows into allocated region below this struct */
};

struct soap_blist
{ struct soap_blist *next;
  char *ptr;
  size_t size;
};

struct soap_array
{ void *__ptr;
  int __size;
};

/* pointer serialization management */
struct soap_plist
{ struct soap_plist *next;
  const void *ptr;
  const struct soap_array *array;
  int type;
  int id;
  char mark1;
  char mark2;
};

/* class allocation list */
struct soap_clist
{ struct soap_clist *next;
  void *ptr;
  int type;
  int size;
  void (*fdelete)(struct soap_clist*);
};

/* id-ref forwarding list */
struct soap_ilist
{ struct soap_ilist *next;
  int type;
  size_t size;
  void *link;
  void *copy;
  struct soap_flist *flist;
  void *ptr;
  unsigned int level;
  char id[1]; /* the actual id string value overflows into allocated region below this struct */
};

struct soap_attribute
{ struct soap_attribute *next;
  short visible;
  char *value;
  size_t size;
  char *ns;
  char name[1]; /* the actual name string overflows into allocated region below this struct */
};

struct soap_cookie
{ struct soap_cookie *next;
  char *name;
  char *value;
  char *domain;
  char *path;
  long expire;		/* client-side: local time to expire; server-side: seconds to expire */
  unsigned int version;
  short secure;
  short session;	/* server-side */
  short env;		/* server-side: got cookie from client */
  short modified;	/* server-side: client cookie was modified */
};

struct soap_dom_attribute
{ struct soap_dom_attribute *next;
  const char *nstr;
  char *name;
  char *data;
  wchar_t *wide;
  struct soap *soap;
#ifdef __cplusplus
  struct soap_dom_attribute &set(const char *nstr, const char *name);	// set namespace and name
  struct soap_dom_attribute &set(const char *data);		// set data
  void unlink();
  soap_dom_attribute();
  soap_dom_attribute(struct soap *soap);
  soap_dom_attribute(struct soap *soap, const char *nstr, const char *name, const char *data);
  ~soap_dom_attribute();
#endif
};

#ifdef __cplusplus
class soap_dom_iterator
{ public:
  struct soap_dom_element *elt;
  const char *nstr;
  const char *name;
  int type;
  bool operator==(const soap_dom_iterator&) const;
  bool operator!=(const soap_dom_iterator&) const;
  struct soap_dom_element &operator*() const;
  soap_dom_iterator &operator++();
  soap_dom_iterator();
  soap_dom_iterator(struct soap_dom_element*);
  ~soap_dom_iterator();
};
#endif

struct soap_dom_element
{ struct soap_dom_element *next;	/* next sibling */
  struct soap_dom_element *prnt;	/* parent */
  struct soap_dom_element *elts;	/* first child element */
  struct soap_dom_attribute *atts;	/* first child attribute */
  const char *nstr;			/* namespace string */
  char *name;				/* element tag name */
  char *data;				/* element content data (with SOAP_C_UTFSTRING flag set) */
  wchar_t *wide;			/* element content data */
  int type;				/* optional: serialized C/C++ data type */
  void *node;				/* optional: pointer to serialized C/C++ data */
  struct soap *soap;
#ifdef __cplusplus
  typedef soap_dom_iterator iterator;
  struct soap_dom_element &set(const char *nstr, const char *name);
  struct soap_dom_element &set(const char *data);
  struct soap_dom_element &set(void *node, int type);
  struct soap_dom_element &add(struct soap_dom_element*);
  struct soap_dom_element &add(struct soap_dom_element&);
  struct soap_dom_element &add(struct soap_dom_attribute*);
  struct soap_dom_element &add(struct soap_dom_attribute&);
  soap_dom_iterator begin();
  soap_dom_iterator end();
  soap_dom_iterator find(const char *nstr, const char *name);
  soap_dom_iterator find(int type);
  void unlink();
  soap_dom_element();
  soap_dom_element(struct soap *soap);
  soap_dom_element(struct soap *soap, const char *nstr, const char *name);
  soap_dom_element(struct soap *soap, const char *nstr, const char *name, const char *data);
  soap_dom_element(struct soap *soap, const char *nstr, const char *name, void *node, int type);
  ~soap_dom_element();
#endif
};

#if defined(__cplusplus) && !defined(UNDER_CE)
}
extern ostream &operator<<(ostream&, const struct soap_dom_element&);
extern istream &operator>>(istream&, struct soap_dom_element&);
extern "C" {
#endif

struct soap
{ short version;		/* 1 = SOAP1.1 and 2 = SOAP1.2 (set automatically from namespace URI in nsmap table) */
  unsigned int mode;
  unsigned int imode;
  unsigned int omode;
  short copy;			/* 1 = copy of another soap struct */
  const char *float_format;	/* points to user-definable format string for floats (<1024 chars) */
  const char *double_format;	/* points to user-definable format string for doubles (<1024 chars) */
  const char *dime_id_format;	/* points to user-definable format string for integer DIME id (<SOAP_TAGLEN chars) */
  const char *http_version;	/* default = "1.0" */
  const char *http_content;	/* optional custom response content type (with SOAP_FILE) */
  const char *encodingStyle;	/* default = NULL which means that SOAP encoding is used */
  const char *actor;
  int recv_timeout;		/* when > 0, gives socket recv timeout in seconds, < 0 in usec */
  int send_timeout;		/* when > 0, gives socket send timeout in seconds, < 0 in usec */
  int connect_timeout;		/* when > 0, gives socket connect() timeout in seconds, < 0 in usec */
  int accept_timeout;		/* when > 0, gives socket accept() timeout in seconds, < 0 in usec */
  int socket_flags;		/* socket recv() and send() flags, e.g. set to MSG_NOSIGNAL to disable sigpipe */
  int connect_flags;		/* connect() SOL_SOCKET sockopt flags, e.g. set to SO_DEBUG to debug socket */
  int bind_flags;		/* bind() SOL_SOCKET sockopt flags, e.g. set to SO_REUSEADDR to enable reuse */
  int accept_flags;		/* accept() SOL_SOCKET sockopt flags */
  struct Namespace *namespaces;	/* Pointer to global namespace mapping table */
  struct Namespace *local_namespaces;	/* Local namespace mapping table */
  struct soap_nlist *nlist;	/* namespace stack */
  struct soap_blist *blist;	/* block allocation stack */
  struct soap_clist *clist;	/* class instance allocation list */
  void *alist;			/* memory allocation list */
  struct soap_ilist *iht[SOAP_IDHASH];
  struct soap_plist *pht[SOAP_PTRHASH];
  struct SOAP_ENV__Header *header;
  struct SOAP_ENV__Fault *fault;
  void *user;			/* to pass user-defined data */
  struct soap_plugin *plugins;	/* linked list of plug-in data */
  char *userid;			/* HTTP Basic authorization userid */
  char *passwd;			/* HTTP Basic authorization passwd */
  int (*fpost)(struct soap*, const char*, const char*, int, const char*, const char*, size_t);
  int (*fget)(struct soap*);
  int (*fposthdr)(struct soap*, const char*, const char*);
  int (*fresponse)(struct soap*, int, size_t);
  int (*fparse)(struct soap*);
  int (*fparsehdr)(struct soap*, const char*, const char*);
  int (*fconnect)(struct soap*, const char*, const char*, int);
  int (*fdisconnect)(struct soap*);
  int (*fopen)(struct soap*, const char*, const char*, int);
  int (*faccept)(struct soap*, int, struct sockaddr*, int *n);
  int (*fclose)(struct soap*);
  int (*fsend)(struct soap*, const char*, size_t);
  size_t (*frecv)(struct soap*, char*, size_t);
  int (*fprepare)(struct soap*, const char*, size_t);
  int (*fignore)(struct soap*, const char*);
  void *(*fplugin)(struct soap*, const char*);
  void *(*fdimereadopen)(struct soap*, void*, const char*, const char*, const char*);
  void *(*fdimewriteopen)(struct soap*, const char*, const char*, const char*);
  void (*fdimereadclose)(struct soap*, void*);
  void (*fdimewriteclose)(struct soap*, void*);
  size_t (*fdimeread)(struct soap*, void*, char*, size_t);
  int (*fdimewrite)(struct soap*, void*, const char*, size_t);
  int master;
  int socket;
#if defined(__cplusplus) && !defined(UNDER_CE)
  ostream *os;
  istream *is;
#else
  void *os;	/* preserve alignment */
  void *is;	/* preserve alignment */
#endif
#ifndef UNDER_CE
  int sendfd;
  int recvfd;
#else
  FILE *sendfd;
  FILE *recvfd;
  char errorstr[256];
  wchar_t werrorstr[256];
#endif
  size_t bufidx;
  size_t buflen;
  wchar ahead;
  short cdata;
  short body;
  unsigned int level;
  size_t count;		/* message length counter */
  size_t length;	/* message length as set by HTTP header */
  char *labbuf;		/* look-aside buffer */
  size_t lablen;	/* look-aside buffer allocated length */
  size_t labidx;	/* look-aside buffer index to available part */
  char buf[SOAP_BUFLEN];/* send and receive buffer */
  char msgbuf[1024];	/* output buffer for (error) messages <=1024 bytes */
  char tmpbuf[1024];	/* output buffer for HTTP headers, simpleType values, attribute names, and DIME >=1024 bytes */
  char tag[SOAP_TAGLEN];
  char id[SOAP_TAGLEN];
  char href[SOAP_TAGLEN];
  char type[SOAP_TAGLEN];
  char arrayType[SOAP_TAGLEN];
  char arraySize[SOAP_TAGLEN];
  char arrayOffset[SOAP_TAGLEN];
  short other;
  short root;
  short position;
  int positions[SOAP_MAXDIMS];
  struct soap_attribute *attributes;	/* attribute list */
  short encoding;
  short mustUnderstand;
  short null;
  short ns;
  short part;
  short alloced;
  short peeked;
  short dot_net_bug;
  short keep_alive;
  size_t chunksize;
  size_t chunkbuflen;
  char endpoint[SOAP_TAGLEN];
  char path[SOAP_TAGLEN];
  char host[SOAP_TAGLEN];
  char *action;
  int port;
  unsigned int max_keep_alive;
  const char *proxy_host;	/* Proxy Server host name */
  int proxy_port;		/* Proxy Server port (default = 8080) */
  const char *proxy_userid;	/* Proxy Authorization user name */
  const char *proxy_passwd;	/* Proxy Authorization password */
  int status;			/* -1 when request, else error code to be returned by server */
  int error;
  int errmode;
  int errnum;
  unsigned long idnum;
  unsigned long ip;
  size_t dime_count;
  int dime_flags;
  size_t dime_size;
  size_t dime_chunksize;
  size_t dime_buflen;
  char *dime_ptr;
  char *dime_id;
  char *dime_type;
  char *dime_options;
  struct soap_dom_element *dom;
#ifndef WITH_LEAN
  const char *logfile[SOAP_MAXLOGS];
  FILE *fdebug[SOAP_MAXLOGS];
  struct soap_cookie *cookies;
  const char *cookie_domain;
  const char *cookie_path;
  int cookie_max;
#endif
#ifdef WITH_OPENSSL
  int (*fsslauth)(struct soap*);
  int (*fsslverify)(int, X509_STORE_CTX*);
  BIO *bio;
  SSL *ssl;
  SSL_CTX *ctx;
  short require_server_auth;
  short require_client_auth;
  short rsa;			/* when set, use RSA instead of DH */
  const char *keyfile;
  const char *password;
  const char *dhfile;
  const char *cafile;
  const char *capath;
  const char *randfile;
  SSL_SESSION *session;
  char session_host[SOAP_TAGLEN];
  int session_port;
#endif
#ifdef WITH_ZLIB
  short zlib_state;		/* SOAP_ZLIB_NONE, SOAP_ZLIB_DEFLATE, or SOAP_ZLIB_INFLATE */
  short zlib_in;		/* SOAP_ZLIB_NONE, SOAP_ZLIB_DEFLATE, or SOAP_ZLIB_GZIP */
  short zlib_out;		/* SOAP_ZLIB_NONE, SOAP_ZLIB_DEFLATE, or SOAP_ZLIB_GZIP */
  z_stream d_stream;		/* decompression stream */
  char z_buf[SOAP_BUFLEN];	/* buffer */
  size_t z_buflen;
  unsigned short z_level;	/* compression level to be used (0=none, 1=fast to 9=best) */
  unsigned long z_crc;		/* internal gzip crc */
  float z_ratio_in;		/* detected compression ratio compressed_length/length of inbound message */
  float z_ratio_out;		/* detected compression ratio compressed_length/length of outbound message */
#endif
#ifdef PALM
   UInt16 stdLibNum;
   UInt16 stdLib2Num;
   UInt16 stdLib3Num;
   UInt16 genLibNum;
   Err fH_errno;
   Err fErrno;
   Int32 timeout;
   NetHostInfoBufType hostInfo;
   UInt16 socketLibNum;
#endif
/* WR[ */
#ifdef WMW_RPM_IO
  void *rpmreqid;
#endif /* WMW_RPM_IO */
/* ]WR */
};

struct soap_code_map
{ long code;
  const char *string;
};

/* forwarding list for container elements */
struct soap_flist
{ struct soap_flist *next;
  int type;
  void *ptr;
  unsigned int level;
  void (*finsert)(struct soap*, int, void*, void*);
};

struct soap_plugin
{ struct soap_plugin *next;
  const char *id;
  void *data;
  int (*fcopy)(struct soap *soap, struct soap_plugin *dst, struct soap_plugin *src);
  void (*fdelete)(struct soap *soap, struct soap_plugin *p); /* should delete fields of plugin only and not free(p) */
};

#ifndef WITH_NONAMESPACES
extern SOAP_NMAC struct Namespace namespaces[];
#endif

#ifdef HAVE_STRRCHR
 #define soap_strrchr(s, t) strrchr(s, t)
#else
 SOAP_FMAC1 char* SOAP_FMAC2 soap_strrchr(const char *s, int t);
#endif

#ifdef HAVE_STRTOL
 #define soap_strtol(s, t, b) strtol(s, t, b)
#else
 SOAP_FMAC1 long SOAP_FMAC2 soap_strtol(const char *s, char **t, int b);
#endif

#ifdef HAVE_STRTOUL
 #define soap_strtoul(s, t, b) strtoul(s, t, b)
#else
 SOAP_FMAC1 unsigned long SOAP_FMAC2 soap_strtoul(const char *s, char **t, int b);
#endif

SOAP_FMAC1 void SOAP_FMAC2 soap_fault(struct soap*);
SOAP_FMAC1 const char** SOAP_FMAC2 soap_faultcode(struct soap*);
SOAP_FMAC1 const char** SOAP_FMAC2 soap_faultstring(struct soap*);
SOAP_FMAC1 const char** SOAP_FMAC2 soap_faultdetail(struct soap*);
SOAP_FMAC1 void SOAP_FMAC2 soap_serializeheader(struct soap*);
SOAP_FMAC1 int SOAP_FMAC2 soap_putheader(struct soap*);
SOAP_FMAC1 int SOAP_FMAC2 soap_getheader(struct soap*);
SOAP_FMAC1 void SOAP_FMAC2 soap_serializefault(struct soap*);
SOAP_FMAC1 int SOAP_FMAC2 soap_putfault(struct soap*);
SOAP_FMAC1 int SOAP_FMAC2 soap_getfault(struct soap*);

SOAP_FMAC1 int SOAP_FMAC2 soap_poll(struct soap*);
SOAP_FMAC1 int SOAP_FMAC2 soap_connect_command(struct soap*, int, const char*, const char*);
SOAP_FMAC1 int SOAP_FMAC2 soap_connect(struct soap*, const char*, const char*);
SOAP_FMAC1 int SOAP_FMAC2 soap_bind(struct soap*, const char*, int, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_accept(struct soap*);
SOAP_FMAC1 int SOAP_FMAC2 soap_ssl_accept(struct soap*);

SOAP_FMAC1 int SOAP_FMAC2 soap_ssl_server_context(struct soap*, unsigned short, const char*, const char*, const char*, const char*, const char*, const char*, const char*);
SOAP_FMAC1 int SOAP_FMAC2 soap_ssl_client_context(struct soap*, unsigned short, const char*, const char*, const char*, const char*, const char*);

SOAP_FMAC1 int SOAP_FMAC2 soap_puthttphdr(struct soap*, int status, size_t count);

SOAP_FMAC1 int SOAP_FMAC2 soap_hash(const char*);
SOAP_FMAC1 void SOAP_FMAC2 soap_set_endpoint(struct soap*, const char*);
SOAP_FMAC1 int SOAP_FMAC2 soap_flush_raw(struct soap*, const char*, size_t);
SOAP_FMAC1 int SOAP_FMAC2 soap_flush(struct soap*);
SOAP_FMAC1 wchar SOAP_FMAC2 soap_get(struct soap*);
SOAP_FMAC1 wchar SOAP_FMAC2 soap_getchar(struct soap*);
SOAP_FMAC1 int SOAP_FMAC2 soap_tag_cmp(const char*, const char*);
SOAP_FMAC1 void SOAP_FMAC2 soap_set_fault(struct soap*);
SOAP_FMAC1 int SOAP_FMAC2 soap_sender_fault(struct soap*, const char*, const char*);
SOAP_FMAC1 int SOAP_FMAC2 soap_receiver_fault(struct soap*, const char*, const char*);
SOAP_FMAC1 int SOAP_FMAC2 soap_set_sender_error(struct soap*, const char*, const char*, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_set_receiver_error(struct soap*, const char*, const char*, int);

SOAP_FMAC1 int SOAP_FMAC2 soap_send_raw(struct soap*, const char*, size_t);
SOAP_FMAC1 int SOAP_FMAC2 soap_recv_raw(struct soap*);
SOAP_FMAC1 int SOAP_FMAC2 soap_send(struct soap*, const char*);
SOAP_FMAC1 int SOAP_FMAC2 soap_recv(struct soap*);

SOAP_FMAC1 int SOAP_FMAC2 soap_pututf8(struct soap*, unsigned long);
SOAP_FMAC1 wchar SOAP_FMAC2 soap_getutf8(struct soap*);

SOAP_FMAC1 int SOAP_FMAC2 soap_putbase64(struct soap*, const unsigned char*, int);
SOAP_FMAC1 unsigned char* SOAP_FMAC2 soap_getbase64(struct soap*, int*, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_puthex(struct soap*, const unsigned char*, int);
SOAP_FMAC1 unsigned char* SOAP_FMAC2 soap_gethex(struct soap*, int*);


SOAP_FMAC1 struct soap_ilist* SOAP_FMAC2 soap_lookup(struct soap*, const char*);
SOAP_FMAC1 struct soap_ilist* SOAP_FMAC2 soap_enter(struct soap*, const char*);
SOAP_FMAC1 void SOAP_FMAC2 soap_resolve_ptr(struct soap_ilist*);
SOAP_FMAC1 int SOAP_FMAC2 soap_resolve(struct soap*);

SOAP_FMAC1 int SOAP_FMAC2 soap_pointer_lookup(struct soap*, const void *p, int t, struct soap_plist**);
SOAP_FMAC1 int SOAP_FMAC2 soap_array_pointer_lookup(struct soap*, const void *p, const struct soap_array *a, int n, int t, struct soap_plist**);
SOAP_FMAC1 int SOAP_FMAC2 soap_pointer_lookup_id(struct soap*, void *p, int t, struct soap_plist**);
SOAP_FMAC1 int SOAP_FMAC2 soap_pointer_enter(struct soap*, const void *p, int t, struct soap_plist**);
SOAP_FMAC1 int SOAP_FMAC2 soap_array_pointer_enter(struct soap*, const void *p, const struct soap_array *a, int t, struct soap_plist**);

SOAP_FMAC1 int SOAP_FMAC2 soap_embed_element(struct soap *soap, const void *p, const char *tag, int type);
SOAP_FMAC1 int SOAP_FMAC2 soap_embed_array(struct soap *soap, const void *p, const struct soap_array *a, int n, const char *tag, int type);

SOAP_FMAC1 void SOAP_FMAC2 soap_begin_count(struct soap*);
SOAP_FMAC1 int SOAP_FMAC2 soap_begin_send(struct soap*);
SOAP_FMAC1 int SOAP_FMAC2 soap_end_send(struct soap*);

SOAP_FMAC1 void SOAP_FMAC2 soap_embedded(struct soap*, const void *p, int t);
SOAP_FMAC1 int SOAP_FMAC2 soap_reference(struct soap*, const void *p, int t);
SOAP_FMAC1 int SOAP_FMAC2 soap_array_reference(struct soap*, const void *p, const struct soap_array *a, int n, int t);
SOAP_FMAC1 int SOAP_FMAC2 soap_embedded_id(struct soap*, int id, const void *p, int t);
SOAP_FMAC1 int SOAP_FMAC2 soap_is_embedded(struct soap*, struct soap_plist*);
SOAP_FMAC1 int SOAP_FMAC2 soap_is_single(struct soap*, struct soap_plist*);
SOAP_FMAC1 int SOAP_FMAC2 soap_is_multi(struct soap*, struct soap_plist*);
SOAP_FMAC1 void SOAP_FMAC2 soap_set_embedded(struct soap*, struct soap_plist*);

SOAP_FMAC1 const struct soap_code_map* SOAP_FMAC2 soap_code(const struct soap_code_map*, const char *str);
SOAP_FMAC1 long SOAP_FMAC2 soap_int_code(const struct soap_code_map*, const char *str, long other);
SOAP_FMAC1 const char* SOAP_FMAC2 soap_str_code(const struct soap_code_map*, long code);

SOAP_FMAC1 int SOAP_FMAC2 soap_getline(struct soap*, char*, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_begin_recv(struct soap*);
SOAP_FMAC1 int SOAP_FMAC2 soap_end_recv(struct soap*);

SOAP_FMAC1 void* SOAP_FMAC2 soap_malloc(struct soap*, size_t);
SOAP_FMAC1 void SOAP_FMAC2 soap_dealloc(struct soap*, void*);
SOAP_FMAC1 struct soap_clist * SOAP_FMAC2 soap_link(struct soap*, void*, int, int, void (*fdelete)(struct soap_clist*));
SOAP_FMAC1 void SOAP_FMAC2 soap_unlink(struct soap*, const void*);
SOAP_FMAC1 void SOAP_FMAC2 soap_free(struct soap*);

SOAP_FMAC1 int SOAP_FMAC2 soap_lookup_type(struct soap*, const char *id);

SOAP_FMAC1 void* SOAP_FMAC2 soap_id_lookup(struct soap*, const char *id, void **p, int t, size_t n, unsigned int k);
SOAP_FMAC1 void* SOAP_FMAC2 soap_id_forward(struct soap*, const char *id, void *p, int t, size_t n);
SOAP_FMAC1 void* SOAP_FMAC2 soap_id_enter(struct soap*, const char *id, void *p, int t, size_t n, int k);

SOAP_FMAC1 int SOAP_FMAC2 soap_size(const int *, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_getoffsets(const char *, const int *, int *, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_getsize(const char *, const char *, int *);
SOAP_FMAC1 int SOAP_FMAC2 soap_getsizes(const char *, int *, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_getposition(const char *, int *);

SOAP_FMAC1 char* SOAP_FMAC2 soap_putsize(struct soap*, const char *, int);
SOAP_FMAC1 char* SOAP_FMAC2 soap_putsizesoffsets(struct soap*, const char *, const int *, const int *, int);
SOAP_FMAC1 char* SOAP_FMAC2 soap_putsizes(struct soap*, const char *, const int *, int);
SOAP_FMAC1 char* SOAP_FMAC2 soap_putoffset(struct soap*, int);
SOAP_FMAC1 char* SOAP_FMAC2 soap_putoffsets(struct soap*, const int *, int);
 
SOAP_FMAC1 int SOAP_FMAC2 soap_closesock(struct soap*);

SOAP_FMAC1 struct soap *SOAP_FMAC2 soap_new(void);
SOAP_FMAC1 struct soap *SOAP_FMAC2 soap_new1(int);
SOAP_FMAC1 struct soap *SOAP_FMAC2 soap_new2(int, int);
SOAP_FMAC1 struct soap *SOAP_FMAC2 soap_copy(struct soap*);
SOAP_FMAC1 struct soap *SOAP_FMAC2 soap_copy_context(struct soap*,struct soap*);
SOAP_FMAC1 void SOAP_FMAC2 soap_init(struct soap*);
SOAP_FMAC1 void SOAP_FMAC2 soap_init1(struct soap*, int);
SOAP_FMAC1 void SOAP_FMAC2 soap_init2(struct soap*, int, int);
SOAP_FMAC1 void SOAP_FMAC2 soap_done(struct soap*);
SOAP_FMAC1 void SOAP_FMAC2 soap_cleanup(struct soap*);
SOAP_FMAC1 void SOAP_FMAC2 soap_begin(struct soap*);
SOAP_FMAC1 void SOAP_FMAC2 soap_end(struct soap*);
SOAP_FMAC1 void SOAP_FMAC2 soap_delete(struct soap*, void*);

#ifndef WITH_LEAN
SOAP_FMAC1 void SOAP_FMAC2 soap_set_recv_logfile(struct soap*, const char*);
SOAP_FMAC1 void SOAP_FMAC2 soap_set_sent_logfile(struct soap*, const char*);
SOAP_FMAC1 void SOAP_FMAC2 soap_set_test_logfile(struct soap*, const char*);
SOAP_FMAC1 void SOAP_FMAC2 soap_close_logfiles(struct soap*);
SOAP_FMAC1 void SOAP_FMAC2 soap_open_logfile(struct soap*, int);
#endif

SOAP_FMAC1 char* SOAP_FMAC2 soap_value(struct soap*);

SOAP_FMAC1 wchar SOAP_FMAC2 soap_advance(struct soap*);
SOAP_FMAC1 wchar SOAP_FMAC2 soap_skip(struct soap*);

SOAP_FMAC1 int SOAP_FMAC2 soap_match_tag(struct soap*, const char*, const char *);
SOAP_FMAC1 int SOAP_FMAC2 soap_match_array(struct soap*, const char*);

SOAP_FMAC1 int SOAP_FMAC2 soap_element(struct soap*, const char*, int, const char*);
SOAP_FMAC1 int SOAP_FMAC2 soap_element_begin_out(struct soap*, const char *tag, int id, const char *type);
SOAP_FMAC1 int SOAP_FMAC2 soap_array_begin_out(struct soap*, const char *tag, int id, const char *type, const char *offset);
SOAP_FMAC1 int SOAP_FMAC2 soap_element_ref(struct soap*, const char *tag, int id, int href);
SOAP_FMAC1 int SOAP_FMAC2 soap_element_href(struct soap*, const char *tag, int id, const char *href);
SOAP_FMAC1 int SOAP_FMAC2 soap_element_null(struct soap*, const char *tag, int id, const char *type);
SOAP_FMAC1 int SOAP_FMAC2 soap_element_result(struct soap*, const char *tag);
SOAP_FMAC1 int SOAP_FMAC2 soap_element_end_out(struct soap*, const char *tag);
SOAP_FMAC1 int SOAP_FMAC2 soap_element_start_end_out(struct soap*, const char *tag);

SOAP_FMAC1 int SOAP_FMAC2 soap_attribute(struct soap*, const char*, const char*);

SOAP_FMAC1 int SOAP_FMAC2 soap_element_begin_in(struct soap*, const char *tag);

SOAP_FMAC1 int SOAP_FMAC2 soap_element_end_in(struct soap*, const char *tag);

SOAP_FMAC1 int SOAP_FMAC2 soap_peek_element(struct soap*);

SOAP_FMAC1 void SOAP_FMAC2 soap_retry(struct soap*);
SOAP_FMAC1 void SOAP_FMAC2 soap_revert(struct soap*);

SOAP_FMAC1 char* SOAP_FMAC2 soap_strdup(struct soap*, const char*);

SOAP_FMAC1 int SOAP_FMAC2 soap_string_out(struct soap*, const char *s, int flag);
SOAP_FMAC1 char* SOAP_FMAC2 soap_string_in(struct soap*, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_wstring_out(struct soap*, const wchar_t *s, int flag);
SOAP_FMAC1 wchar_t* SOAP_FMAC2 soap_wstring_in(struct soap*, int);

SOAP_FMAC1 int SOAP_FMAC2 soap_match_namespace(struct soap*, const char *, const char*, int n1, int n2);
SOAP_FMAC1 const char* SOAP_FMAC2 soap_default_namespace(struct soap*);

SOAP_FMAC1 int SOAP_FMAC2 soap_set_namespaces(struct soap*, struct Namespace*);

SOAP_FMAC1 void SOAP_FMAC2 soap_pop_namespace(struct soap*);
SOAP_FMAC1 int SOAP_FMAC2 soap_push_namespace(struct soap*, const char *,const char *);
SOAP_FMAC1 int SOAP_FMAC2 soap_push_default_namespace(struct soap*, const char *, int);

SOAP_FMAC1 int SOAP_FMAC2 soap_new_block(struct soap*);
SOAP_FMAC1 void* SOAP_FMAC2 soap_push_block(struct soap*, size_t);
SOAP_FMAC1 void SOAP_FMAC2 soap_pop_block(struct soap*);
SOAP_FMAC1 size_t SOAP_FMAC2 soap_size_block(struct soap*, size_t);
SOAP_FMAC1 char* SOAP_FMAC2 soap_first_block(struct soap*);
SOAP_FMAC1 char* SOAP_FMAC2 soap_next_block(struct soap*);
SOAP_FMAC1 size_t SOAP_FMAC2 soap_block_size(struct soap*);
SOAP_FMAC1 char* SOAP_FMAC2 soap_save_block(struct soap*, char*);
SOAP_FMAC1 char* SOAP_FMAC2 soap_store_block(struct soap*, char*);
SOAP_FMAC1 void SOAP_FMAC2 soap_end_block(struct soap*);

SOAP_FMAC1 int SOAP_FMAC2 soap_envelope_begin_out(struct soap*);
SOAP_FMAC1 int soap_envelope_end_out(struct soap*);

SOAP_FMAC1 int SOAP_FMAC2 soap_envelope_begin_in(struct soap*);
SOAP_FMAC1 int SOAP_FMAC2 soap_envelope_end_in(struct soap*);

SOAP_FMAC1 int SOAP_FMAC2 soap_body_begin_out(struct soap*);
SOAP_FMAC1 int SOAP_FMAC2 soap_body_end_out(struct soap*);

SOAP_FMAC1 int SOAP_FMAC2 soap_body_begin_in(struct soap*);
SOAP_FMAC1 int SOAP_FMAC2 soap_body_end_in(struct soap*);

SOAP_FMAC1 int SOAP_FMAC2 soap_recv_header(struct soap*);

SOAP_FMAC1 int SOAP_FMAC2 soap_response(struct soap*, int);

SOAP_FMAC1 int SOAP_FMAC2 soap_send_fault(struct soap*);

SOAP_FMAC1 int SOAP_FMAC2 soap_recv_fault(struct soap*);

SOAP_FMAC1 void SOAP_FMAC2 soap_print_fault(struct soap*, FILE*);
SOAP_FMAC1 void SOAP_FMAC2 soap_print_fault_location(struct soap*, FILE*);

SOAP_FMAC1 int SOAP_FMAC2 soap_s2byte(struct soap*, const char*, char*);
SOAP_FMAC1 int SOAP_FMAC2 soap_s2short(struct soap*, const char*, short*);
SOAP_FMAC1 int SOAP_FMAC2 soap_s2int(struct soap*, const char*, int*);
SOAP_FMAC1 int SOAP_FMAC2 soap_s2long(struct soap*, const char*, long*);
SOAP_FMAC1 int SOAP_FMAC2 soap_s2LONG64(struct soap*, const char*, LONG64*);
SOAP_FMAC1 int SOAP_FMAC2 soap_s2float(struct soap*, const char*, float*);
SOAP_FMAC1 int SOAP_FMAC2 soap_s2double(struct soap*, const char*, double*);
SOAP_FMAC1 int SOAP_FMAC2 soap_s2unsignedByte(struct soap*, const char*, unsigned char*);
SOAP_FMAC1 int SOAP_FMAC2 soap_s2unsignedShort(struct soap*, const char*, unsigned short*);
SOAP_FMAC1 int SOAP_FMAC2 soap_s2unsignedInt(struct soap*, const char*, unsigned int*);
SOAP_FMAC1 int SOAP_FMAC2 soap_s2unsignedLong(struct soap*, const char*, unsigned long*);
SOAP_FMAC1 int SOAP_FMAC2 soap_s2ULONG64(struct soap*, const char*, ULONG64*);
SOAP_FMAC1 int SOAP_FMAC2 soap_s2dateTime(struct soap*, const char*, time_t*);
SOAP_FMAC1 int SOAP_FMAC2 soap_s2string(struct soap*, const char*, char**);
SOAP_FMAC1 int SOAP_FMAC2 soap_s2QName(struct soap*, const char*, char**);
SOAP_FMAC1 int SOAP_FMAC2 soap_s2base64(struct soap*, const unsigned char*, char*, size_t);

SOAP_FMAC1 const char* SOAP_FMAC2 soap_byte2s(struct soap*, char);
SOAP_FMAC1 const char* SOAP_FMAC2 soap_short2s(struct soap*, short);
SOAP_FMAC1 const char* SOAP_FMAC2 soap_int2s(struct soap*, int);
SOAP_FMAC1 const char* SOAP_FMAC2 soap_long2s(struct soap*, long);
SOAP_FMAC1 const char* SOAP_FMAC2 soap_LONG642s(struct soap*, LONG64);
SOAP_FMAC1 const char* SOAP_FMAC2 soap_float2s(struct soap*, float);
SOAP_FMAC1 const char* SOAP_FMAC2 soap_double2s(struct soap*, double);
SOAP_FMAC1 const char* SOAP_FMAC2 soap_unsignedByte2s(struct soap*, unsigned char);
SOAP_FMAC1 const char* SOAP_FMAC2 soap_unsignedShort2s(struct soap*, unsigned short);
SOAP_FMAC1 const char* SOAP_FMAC2 soap_unsignedInt2s(struct soap*, unsigned int);
SOAP_FMAC1 const char* SOAP_FMAC2 soap_unsignedLong2s(struct soap*, unsigned long);
SOAP_FMAC1 const char* SOAP_FMAC2 soap_ULONG642s(struct soap*, ULONG64);
SOAP_FMAC1 const char* SOAP_FMAC2 soap_dateTime2s(struct soap*, time_t);
SOAP_FMAC1 const char* SOAP_FMAC2 soap_QName2s(struct soap*, const char*);
SOAP_FMAC1 const char* SOAP_FMAC2 soap_base642s(struct soap*, const char*, char*, size_t, size_t*);

SOAP_FMAC1 int* SOAP_FMAC2 soap_inint(struct soap*, const char *tag, int *p, const char *, int);
SOAP_FMAC1 char* SOAP_FMAC2 soap_inbyte(struct soap*, const char *tag, char *p, const char *, int);
SOAP_FMAC1 long* SOAP_FMAC2 soap_inlong(struct soap*, const char *tag, long *p, const char *, int);
SOAP_FMAC1 LONG64* SOAP_FMAC2 soap_inLONG64(struct soap*, const char *tag, LONG64 *p, const char *, int);
SOAP_FMAC1 short* SOAP_FMAC2 soap_inshort(struct soap*, const char *tag, short *p, const char *, int);
SOAP_FMAC1 float* SOAP_FMAC2 soap_infloat(struct soap*, const char *tag, float *p, const char *, int);
SOAP_FMAC1 double* SOAP_FMAC2 soap_indouble(struct soap*, const char *tag, double *p, const char *, int);
SOAP_FMAC1 unsigned char* SOAP_FMAC2 soap_inunsignedByte(struct soap*, const char *tag, unsigned char *p, const char *, int);
SOAP_FMAC1 unsigned short* SOAP_FMAC2 soap_inunsignedShort(struct soap*, const char *tag, unsigned short *p, const char *, int);
SOAP_FMAC1 unsigned int* SOAP_FMAC2 soap_inunsignedInt(struct soap*, const char *tag, unsigned int *p, const char *, int);
SOAP_FMAC1 unsigned long* SOAP_FMAC2 soap_inunsignedLong(struct soap*, const char *tag, unsigned long *p, const char *, int);
SOAP_FMAC1 ULONG64* SOAP_FMAC2 soap_inULONG64(struct soap*, const char *tag, ULONG64 *p, const char *, int);
SOAP_FMAC1 time_t* SOAP_FMAC2 soap_indateTime(struct soap*, const char *tag, time_t *p, const char *, int);
SOAP_FMAC1 char** SOAP_FMAC2 soap_instring(struct soap*, const char *tag, char **p, const char *, int, int);
SOAP_FMAC1 wchar_t** SOAP_FMAC2 soap_inwstring(struct soap*, const char *tag, wchar_t **p, const char *, int);
SOAP_FMAC1 char** SOAP_FMAC2 soap_inliteral(struct soap*, const char *tag, char **p);
SOAP_FMAC1 wchar_t** SOAP_FMAC2 soap_inwliteral(struct soap*, const char *tag, wchar_t **p);

SOAP_FMAC1 int SOAP_FMAC2 soap_outbyte(struct soap*, const char *tag, int id, const char *p, const char *, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_outshort(struct soap*, const char *tag, int id, const short *p, const char *, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_outint(struct soap*, const char *tag, int id, const int *p, const char *, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_outlong(struct soap*, const char *tag, int id, const long *p, const char *, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_outLONG64(struct soap*, const char *tag, int id, const LONG64 *p, const char *, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_outfloat(struct soap*, const char *tag, int id, const float *p, const char *, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_outdouble(struct soap*, const char *tag, int id, const double *p, const char *, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_outunsignedByte(struct soap*, const char *tag, int id, const unsigned char *p, const char *, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_outunsignedShort(struct soap*, const char *tag, int id, const unsigned short *p, const char *, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_outunsignedInt(struct soap*, const char *tag, int id, const unsigned int *p, const char *, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_outunsignedLong(struct soap*, const char *tag, int id, const unsigned long *p, const char *, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_outULONG64(struct soap*, const char *tag, int id, const ULONG64 *p, const char *, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_outdateTime(struct soap*, const char *tag, int id, const time_t *p, const char *, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_outstring(struct soap*, const char *tag, int id, char *const*p, const char *, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_outwstring(struct soap*, const char *tag, int id, wchar_t *const*p, const char *, int);
SOAP_FMAC1 int SOAP_FMAC2 soap_outliteral(struct soap*, const char *tag, char *const*p);
SOAP_FMAC1 int SOAP_FMAC2 soap_outwliteral(struct soap*, const char *tag, wchar_t *const*p);

#ifndef WITH_LEANER
SOAP_FMAC1 void SOAP_FMAC2 soap_set_attached(struct soap*, struct soap_plist*, const char*, const char*, const char*, size_t);
SOAP_FMAC1 int SOAP_FMAC2 soap_move(struct soap*, long);
SOAP_FMAC1 size_t SOAP_FMAC2 soap_tell(struct soap*);
SOAP_FMAC1 char* SOAP_FMAC2 soap_dime_option(struct soap*, unsigned short, const char*);
SOAP_FMAC1 int SOAP_FMAC2 soap_getdimehdr(struct soap*);
SOAP_FMAC1 int SOAP_FMAC2 soap_getdime(struct soap*);
SOAP_FMAC1 int SOAP_FMAC2 soap_putdimehdr(struct soap*);
SOAP_FMAC1 int SOAP_FMAC2 soap_putdime(struct soap*, int, char*, char*, char*, void*, size_t);
#endif

SOAP_FMAC1 int SOAP_FMAC2 soap_register_plugin_arg(struct soap*, int (*fcreate)(struct soap*, struct soap_plugin*, void*), void*);
SOAP_FMAC1 void* SOAP_FMAC2 soap_lookup_plugin(struct soap*, const char*);

SOAP_FMAC1 struct soap_attribute * SOAP_FMAC2 soap_attr(struct soap *soap, const char *name);
SOAP_FMAC1 const char* SOAP_FMAC2 soap_attr_value(struct soap *soap, const char *name);
SOAP_FMAC1 int SOAP_FMAC2 soap_set_attr(struct soap *soap, const char *name, const char *value);
SOAP_FMAC1 void SOAP_FMAC2 soap_clr_attr(struct soap *soap);

#ifdef WITH_COOKIES
SOAP_FMAC1 int SOAP_FMAC2 soap_encode_cookie(const char*, char*, int);
SOAP_FMAC1 const char* SOAP_FMAC2 soap_decode_cookie(char*, int, const char*);
SOAP_FMAC1 extern struct soap_cookie* SOAP_FMAC2 soap_set_cookie(struct soap*, const char*, const char*, const char*, const char*);
SOAP_FMAC1 extern struct soap_cookie* SOAP_FMAC2 soap_cookie(struct soap*, const char*, const char*, const char*);
SOAP_FMAC1 extern char* SOAP_FMAC2 soap_cookie_value(struct soap*, const char*, const char*, const char*);
SOAP_FMAC1 extern long SOAP_FMAC2 soap_cookie_expire(struct soap*, const char*, const char*, const char*);
SOAP_FMAC1 extern int SOAP_FMAC2 soap_set_cookie_expire(struct soap*, const char*, long, const char*, const char*);
SOAP_FMAC1 extern int SOAP_FMAC2 soap_set_cookie_session(struct soap*, const char*, const char*, const char*);
SOAP_FMAC1 extern int SOAP_FMAC2 soap_clr_cookie_session(struct soap*, const char*, const char*, const char*);
SOAP_FMAC1 extern void SOAP_FMAC2 soap_clr_cookie(struct soap*, const char*, const char*, const char*);
SOAP_FMAC1 extern int SOAP_FMAC2 soap_getenv_cookies(struct soap*);
SOAP_FMAC1 extern struct soap_cookie* SOAP_FMAC2 soap_copy_cookies(struct soap*);
SOAP_FMAC1 extern void SOAP_FMAC2 soap_free_cookies(struct soap*);
#endif

#if defined(PALM) && !defined(NOSHAREDLIB) && !(defined(BUILDING_STDSOAP) || defined(BUILDING_STDSOAP2) || defined(BUILDING_STDLIB) || defined(BUILDING_STDLIB2) || defined(PALM_1) || defined(PALM_2))
# include "palmSharedLib.h"
#endif

#ifdef __cplusplus
}
#endif

#endif

