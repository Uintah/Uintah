/* $Id$ */

/**************************************************************************
 *                                                                        *
 *   Copyright (C) 2000 Grub, Inc.                                        *
 *                                                                        *
 *   This program is free software; you can redistribute it and/or modify *
 *   it under the terms of the GNU General Public License as published by *
 *   the Free Software Foundation; either version 1, or (at your option)  *
 *   any later version.                                                   *
 *                                                                        *
 *   This program is distributed in the hope that it will be useful,      *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of       *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
 *   GNU General Public License for more details.                         *
 *                                                                        *
 *   You should have received a copy of the GNU General Public License    *
 *   along with this program; if not, write to the Free Software          *
 *   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.            *
 *                                                                        *
 *   Author:  Igor Stojanovski - ozra   (email: ozra@grub.org)            *
 *                                                                        *
 **************************************************************************/

#ifndef _SOCK_H_
#define _SOCK_H_

#ifdef __cplusplus
extern "C" {
#endif

// uncomment or define to add a check-up on wether this thread was cancelled
// #define __PTHREAD_TESTCANCEL

#ifdef __PTHREAD_TESTCANCEL
#include <pthread.h>
#endif

#define SOCK_ERROR -1
#define SOCK_CLOSED -2
#define SOCK_WOULDBLOCK -3

#include <unistd.h>
#include <errno.h>

ssize_t sock_write( int soc, const void *buf, size_t nbyte);
ssize_t sock_read( int soc, void *buf, size_t nbyte);

#endif /* _SOCK_H_ */

#ifdef __cplusplus
}
#endif

