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

#ifndef _BUF_H_
#define _BUF_H_

#ifdef __cplusplus
extern "C" {
#endif

struct bufferer {

	int fd; /* file desc. to read/write from/to */
	char *buf;       /* the buffer to store data */
	int buf_maxsize; /* the size of this buffer */
	int buf_cursize; /* number of bytes used in the buffer storage */
	int buf_curpos;  /* in buf_read() -- number of bytes returned from
	                    the buffer to the caller from the buffer front;
	                    in buf_flush() -- number of bytes written to
	                    the stream, counting from the buffer front */
	int(* iofn)(int, char *, int, void *); /* User defined input (when reading)
	                                  or output function to be called */
};


/* buf_init()
 * Initializes a buffer structure values.  It does NOT allocate any
 * memory -- that is up to the caller to do.
 * Parameters:
 *   bufs -- an allocated bufferer struct to store buffer information
 *   fd -- file descriptor to read to or write from
 *   buf -- allocated memory to be used as a buffer
 *   buf_maxsize -- size of the buffer memory
 *   iofn -- a user defined function to be called any time a buffer is
 *           filled up (when writing), or a buffer to be refilled (reading);
 *           this function's arguments should be file descriptor, character
 *           buffer to read/write, length of the buffer, a void pointer
 *           which will be passed any time read or write is needed.  An error
 *           condition should be designated when these functions return
 *           negative values, which will be returned to the user
 *           for inspection.  A zero return is also a special value, and is
 *           treated exactly as an error condition.
 */
void buf_init( struct bufferer *bufs,
		int fd,
		char *buf,
		int buf_maxsize,
		int(* iofn)(int, char *, int, void *) );


/* buf_refill()
 * This function should be called only by buf_read() when the read buffer is
 * emptied.  It will "erase" the previous contents by doing it.  It will read
 * as much data as it can, until the buf_maxsize is reached.  The arg
 * argument will be passed to the iofn function defined in buf_init().
 * Returns:
 *   the return value of the user's function.  If the user defined function was
 *   non-negative, it will consider that the buffer was filled with that many
 *   bytes.
 */
int buf_refill(	struct bufferer *bufs, void *arg );


/* buf_read()
 * Reads bytes from an initialized input stream into buf, up to len.  If buffer
 * already has data, it will be simply copied into buf; if the buffer is
 * empty, buf_refill() will be called to read new contents.  The arg
 * argument will be passed to the iofn function defined in buf_init().
 * Parameters:
 *   bufs -- initialized bufferer struct
 *   buf -- location to store the data read
 *   len -- length of buf
 * Returns:
 *   If the user's input function was called and returned zero or negative
 *   value, that same value will be returned.  Otherwise, the number of bytes
 *   copied into buf is returned.
 *   If error was returned, you may safely call this function again later to
 *   continue reading from the stream.
 */
int buf_read(	struct bufferer *bufs,
		char *buf,
		int len,
		void *arg );


/* buf_flush()
 * Flushes the contents of the write buffer to the output stream, and resets
 * it to indicate empty buffer.  If an empty buffer is flushed, the operation
 * will have no effect, and function will return success.  If an error
 * occured, you may safely flush again later the remaining contents of the
 * buffer, without loss of any data. (Provided that function will not return
 * error in one of the future calls.)  The arg argument will be passed to the
 * iofn function defined in buf_init().
 * Returns:
 *   1 -- on successful write of all contents of the buffer, which may require
 *        multiple calls to the user's output function until all bytes are
 *        flushed.
 *   zero or negative value -- as returned from the user's function.
 */
int buf_flush( struct bufferer *bufs, void *arg );


/* buf_write()
 * Writes the contents of buf to the output stream using the user's function.
 * It will call buf_flush() to flush the buffer contents when it fills up.  If
 * the user's output function returns an error condition (value <= 0), the buf
 * was not written to the output, and the same error value is returned.  If
 * argument len is greater than the size of the buffer, it will be flushed
 * more than once.  However, if error occurs while flushing part of buf,
 * the error value will be returned and there is no way of checking how many
 * bytes were sent prior to the error.  In such case, I recommend that len is
 * less then or equal to the buffer size.  The arg argument will be passed to the
 * iofn function defined in buf_init().
 * Parameters:
 *   bufs -- the bufferer structure
 *   buf -- the location to write from
 *   len -- the size ot the buffer to write from
 * Returns:
 *   1 -- on successfull write of all bytes from buf, which may or may not
 *        have been flushed to the output.  A call to buf_flush() will
 *        assure that.
 *   otherwise, any zero or negative value that is returned from the output
 *        function while flushing the contents.
 */
int buf_write(	struct bufferer *bufs,
		char *buf,
		int len,
		void *arg );

#ifdef __cplusplus
}
#endif

#endif

