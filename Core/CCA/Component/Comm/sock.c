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

#include "sock.h"

/* sock_write()
 * Returns:
 *   Number of bytes written; or
 *   SOCK_WOULDBLOCK -- if non-blocking I/O used when write() would block
 *   SOCK_ERROR -- if writing to the socked failed
 */
ssize_t sock_write( int soc, const void *buf, size_t nbyte) {
	ssize_t ret = 0;

#ifdef __PTHREAD_TESTCANCEL
	pthread_testcancel();
#endif

write_again:
	ret = write(soc, buf, nbyte);

#ifdef __PTHREAD_TESTCANCEL
	pthread_testcancel();
#endif

	if ( ret < 0 ) {

	    if (errno == EAGAIN || errno == EWOULDBLOCK)
	    {
		/* log EAGAIN, EWOULDBLOCK */
		return SOCK_WOULDBLOCK;
	    } else if (errno == EINTR) {
		/*
		**	EINTR	A signal was caught during the  write  opera-
		**		tion and no data was transferred.
		*/
		/* LOG "Write Socket call interruted - try again\n" */
		goto write_again;
	    } else {
		if (errno == EPIPE)
			{ /* LOG "Write Socket got EPIPE\n" */ }
		return SOCK_ERROR;
	    }
	}

	return ret;
}


/* sock_read()
 * Returns:
 *   Number of bytes read, but never zero; or
 *   SOCK_WOULDBLOCK -- if non-blocking I/O used when read() would block
 *   SOCK_ERROR -- if reading to the socked failed
 *   SOCK_CLOSED -- if reading from a closed connection
 */
ssize_t sock_read( int soc, void *buf, size_t nbyte) {
	ssize_t ret;

#ifdef __PTHREAD_TESTCANCEL
	pthread_testcancel();
#endif

	ret = read(soc, buf, nbyte);

#ifdef __PTHREAD_TESTCANCEL
	pthread_testcancel();
#endif

	if ( ret < 0 ) {

		if (errno==EAGAIN || errno==EWOULDBLOCK)      /* POSIX */
		{
			/* LOG: "Read Socket. WOULD BLOCK fd %d\n" */
			return SOCK_WOULDBLOCK;
		} else if (errno == EPIPE) {
			/* LOG: "Read Socket. got EPIPE\n" */
			goto socketClosed;
		} else if (errno == ECONNRESET) {
			/* LOG: "Read Socket. got ECONNRESET\n" */
			goto socketClosed;
		} else { 			     /* We have a real error */

			return SOCK_ERROR;
		}
	} else if (!ret) {

	socketClosed:
		/* LOG: "Read Socket. FIN received on socket %d\n" */
/* perror(""); printf("FIN ret == %d\n", ret ); */
		return SOCK_CLOSED;
	}

	return ret;
}

