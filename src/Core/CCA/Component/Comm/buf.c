
/**************************************************************************
 *                                                                        *
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
 *                                                                        *
 **************************************************************************/

#include "buf.h"


void buf_init( struct bufferer *bufs,
		int fd,
		char *buf,
		int buf_maxsize,
		int(* iofn)(int, char *, int, void *) )
{
	bufs->fd = fd;
	bufs->buf = buf;
	bufs->buf_maxsize = buf_maxsize;
	bufs->buf_cursize = 0;
	bufs->buf_curpos = 0;
	bufs->iofn = iofn;
}


int buf_refill(	struct bufferer *bufs, void *arg )
{
	int ret;

	/* read from user's read function */
	ret = bufs->buf_cursize = bufs->iofn(
		bufs->fd,
		bufs->buf,
		bufs->buf_maxsize,
		arg
	);

	bufs->buf_curpos = 0;

	if ( bufs->buf_cursize < 0 )
		bufs->buf_cursize = 0;

	return ret;
}


int buf_read(	struct bufferer *bufs,
		char *buf,
		int len,
		void *arg )
{
	int ret;
	int amt_data;

	if ( len < 1 ) return 0;  /* sanity check */

	if ( bufs->buf_cursize == bufs->buf_curpos )
		if ( ( ret = buf_refill( bufs, arg ) ) <= 0 )
			return ret;

	amt_data = ( len > bufs->buf_cursize - bufs->buf_curpos ) ?
		bufs->buf_cursize - bufs->buf_curpos : len;

	memcpy( buf, bufs->buf + bufs->buf_curpos, amt_data );
	bufs->buf_curpos += amt_data;

	return amt_data;
}


int buf_flush( struct bufferer *bufs, void *arg ) {
	int ret = 1;

	if ( bufs->buf_cursize == 0 )
		return 1;

	while ( bufs->buf_curpos < bufs->buf_cursize ) {
		int fret;

		/* write from user's write function */
		fret = bufs->iofn(
			bufs->fd,
			bufs->buf + bufs->buf_curpos,
			bufs->buf_cursize - bufs->buf_curpos,
			arg
		);
		if ( fret <= 0 )
			return fret;

		bufs->buf_curpos += fret;
	}

	ret = bufs->buf_cursize;
	bufs->buf_cursize = 0;
	bufs->buf_curpos = 0;

	return ret;
}


int buf_write(	struct bufferer *bufs,
		char *buf,
		int len,
		void *arg )
{
	int ret, i;

	for ( i = 0; i < len / bufs->buf_maxsize + 1; i++ ) {
		/* cp_size will equal max_size if len is
		 * greater than buf_maxsize and is NOT in the last
		 * iteration of the loop; otherwise, it will be len
		 */
		int cp_size = ( i == len / bufs->buf_maxsize ) ?
				len - bufs->buf_maxsize * i :
				bufs->buf_maxsize;

		if ( cp_size < 0 ) return 1;

		if ( cp_size > bufs->buf_maxsize - bufs->buf_cursize )
			if ( ( ret = buf_flush( bufs, arg ) ) <= 0 )
				return ret;

		memcpy( bufs->buf + bufs->buf_cursize,
			buf + i * bufs->buf_maxsize,
			cp_size
		);
		bufs->buf_cursize += cp_size;
	}

	return 1;
}

