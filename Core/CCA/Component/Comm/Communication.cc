
#include "Communication.h"

using std::string;

/* write_abstract()
 * Used by buffered reads/writes.  This datatype (see buf.h) will call this
 * function any time it needs to write data.  The return codes of this
 * function are then passed back to the caller as return value of buf_write()
 * or buf_flush().
 * Parameters:
 *   fd -- the file descriptor to write to
 *   buf -- the buffer of data to write from
 *   len -- the length of this buffer
 *   arg -- (struct *sock_arg) type; used with select()
 * Returns:
 *   Number of bytes written; or
 *   COMM_TIMEOUT -- if waiting for available write timed out
 *   COMM_SELECTFAILED -- if select() failed; check errno for more info
 *   SOCK_WOULDBLOCK -- if non-blocking I/O used when write() would block
 *   SOCK_ERROR -- if writing to the socked failed
 */
int write_abstract( int fd, char *buf, int len, void *arg ) {
	sock_arg *sock = (sock_arg *)arg;
	int ret;

	if ( arg != NULL && sock->timeout.tv_sec != 0 ) {

#ifdef __PTHREAD_TESTCANCEL
		pthread_testcancel();
#endif

		FD_SET( fd, &sock->set );
		ret = select( fd + 1, NULL, &sock->set, NULL, &sock->timeout );

#ifdef __PTHREAD_TESTCANCEL
		pthread_testcancel();
#endif

		if ( ret == 0 ) {

			/* write file desc. didn't become available
			 * after the specified time elapsed
			 */
			return COMM_TIMEOUT;
		}
		else if ( ret == -1 ) {

			/* select() failed; check errno for reason */
			return COMM_SELECTFAILED;
		}
	}

	ret = (int)sock_write( fd, buf, len );
	return ret;
}


/* read_abstract()
 * Similar to write_abstract(), ths function is called any time a read is
 * needed.
 * Parameters:
 *   fd -- the file descriptor to read from
 *   buf -- the buffer of data to read to
 *   len -- the length of this buffer
 *   arg -- (struct *sock_arg) type; used with select()
 * Returns:
 *   Number of bytes read, but never zero bytes; or
 *   COMM_TIMEOUT -- if waiting for read timed out
 *   COMM_SELECTFAILED -- if select() failed; check errno for more info
 *   SOCK_WOULDBLOCK -- if non-blocking I/O used when read() would block
 *   SOCK_ERROR -- if reading to the socked failed
 *   SOCK_CLOSED -- if reading from a closed connection
 */
int read_abstract( int fd, char *buf, int len, void *arg ) {
	sock_arg *sock = (sock_arg *)arg;
	int ret;

	if ( arg != NULL && sock->timeout.tv_sec != 0 ) {

#ifdef __PTHREAD_TESTCANCEL
		pthread_testcancel();
#endif

		FD_SET( fd, &sock->set );
		ret = select( fd + 1, &sock->set, NULL, NULL, &sock->timeout );

#ifdef __PTHREAD_TESTCANCEL
		pthread_testcancel();
#endif

		if ( ret == 0 ) {

			/* write file desc. didn't become available
			 * after the specified time elapsed
			 */
			return COMM_TIMEOUT;
		}
		else if ( ret == -1 ) {

			/* select() failed; check errno for reason */
			return COMM_SELECTFAILED;
		}
	}

	ret = (int)sock_read( fd, buf, len );
	return ret;
}

int write_abstract_custom( int fd, char *buf, int len, void *arg ) {
	Communication *comm = (Communication *)arg;
	int ret = 0;

	if ( comm->write_fn ) {
		if ( comm->timeout.tv_sec > 0 ) {
			struct timeval timeout = {
				comm->timeout.tv_sec,
				comm->timeout.tv_usec
			};

#ifdef __PTHREAD_TESTCANCEL
			pthread_testcancel();
#endif

			FD_SET( fd, &comm->write_custom_fd_set );
			ret = select( fd + 1, NULL,
				&comm->write_custom_fd_set, NULL,
				&timeout );

#ifdef __PTHREAD_TESTCANCEL
			pthread_testcancel();
#endif

			if ( ret == 0 ) {

				/* write file desc. didn't become available
				 * after the specified time elapsed
				 */
				return COMM_TIMEOUT;
			}
			else if ( ret == -1 ) {

				/* select() failed; check errno for reason */
				return COMM_SELECTFAILED;
			}
		}

		ret = comm->write_fn( fd, buf, len, comm, comm->w_fn_arg );
	}

	return ret;
}

int read_abstract_custom( int fd, char *buf, int len, void *arg ) {
	Communication *comm = (Communication *)arg;
	int ret = 0;

	if ( comm->read_fn ) {
		if ( comm->timeout.tv_sec > 0 ) {
			// must copy the struct because Linux modifies it
			struct timeval timeout = {
				comm->timeout.tv_sec,
				comm->timeout.tv_usec
			};

#ifdef __PTHREAD_TESTCANCEL
			pthread_testcancel();
#endif

			FD_SET( fd, &comm->read_custom_fd_set );
			ret = select( fd + 1, &comm->read_custom_fd_set,
				NULL, NULL,
				&timeout );

#ifdef __PTHREAD_TESTCANCEL
			pthread_testcancel();
#endif

			if ( ret == 0 ) {

				/* write file desc. didn't become available
				 * after the specified time elapsed
				 */
				return COMM_TIMEOUT;
			}
			else if ( ret == -1 ) {

				/* select() failed; check errno for reason */
				return COMM_SELECTFAILED;
			}
		}

		ret = comm->read_fn( fd, buf, len, comm, comm->r_fn_arg );
	}

	return ret;
}

// SOCK_ERROR
int Communication::write( const char *buf, int len ) {
	int ret = 0, total_w = 0;

	if ( w_comm ) {

		ret = w_comm->write( buf, len );
		if ( ret < 0 ) {

			copy_err( w_comm );
			return -1;
		}
	}
	else {
		// initialize the time struct, as Linux modifies it every time
		// select() gets called
		write_arg.timeout.tv_sec  = timeout.tv_sec;
		write_arg.timeout.tv_usec = timeout.tv_usec;

		// ret = bytes written, SOCK_WOULDBLOCK, SOCK_ERROR
		if ( use_custom_io )
			ret = buf_write( &buff_w, (char *)buf, len, (void *)this );
		else
			ret = buf_write( &buff_w, (char *)buf, len, (void *)&write_arg );
		if ( ret < 0 ) {
			if ( ret == SOCK_ERROR && sock_err ) {

				Comm_ERROR( SOCK_ERROR, "Write Socket: got EPIPE" );
				return -1;
			}
			else if ( ret == COMM_TIMEOUT ) {

				Comm_ERROR( COMM_TIMEOUT,
					"Write Socket: connection timed out" );
				return -1;
			}
			else if ( ret == COMM_SELECTFAILED ) {

				Comm_ERROR( COMM_SELECTFAILED, strerror( errno ) );
				return -1;
			}
			else if ( ret == SOCK_WOULDBLOCK && sock_err ) {

				Comm_TRACE( SOCK_WOULDBLOCK,
					"Write Socket: WOULD BLOCK" );
				return -1;
			}
			else if ( use_custom_io ) {

				Comm_ERROR( ret, "Custom I/O error" );
				return -1;
			}
			else {

				Comm_ERROR( COMM_IO_ERROR, "I/O error occured" );
				return -1;
			}
		}
	}

	return 0;
}


int Communication::flush() {
	int ret;

	if ( w_comm ) {

		ret = w_comm->flush();
		if ( ret < 0 ) {

			copy_err( w_comm );
			return -1;
		}
	}
	else {
		// initialize the time struct, as Linux modifies it every time
		// select() gets called
		write_arg.timeout.tv_sec  = timeout.tv_sec;
		write_arg.timeout.tv_usec = timeout.tv_usec;

		if ( use_custom_io )
			ret = buf_flush( &buff_w, (void *)this );
		else
			ret = buf_flush( &buff_w, (void *)&write_arg );
		if ( ret < 0 ) {
			if ( ret == SOCK_ERROR && sock_err ) {

				Comm_ERROR( SOCK_ERROR, "Write Socket: got EPIPE" );
				return -1;
			}
			else if ( ret == COMM_TIMEOUT ) {

				Comm_ERROR( COMM_TIMEOUT,
					"Write Socket: connection timed out" );
				return -1;
			}
			else if ( ret == COMM_SELECTFAILED ) {

				Comm_ERROR( COMM_SELECTFAILED, strerror( errno ) );
				return -1;
			}
			else if ( ret == SOCK_WOULDBLOCK && sock_err ) {

				Comm_TRACE( SOCK_WOULDBLOCK,
					"Write Socket: WOULD BLOCK" );
				return -1;
			}
			else if ( use_custom_io ) {

				Comm_ERROR( ret, "Custom I/O error" );
				return -1;
			}
			else {

				Comm_ERROR( COMM_IO_ERROR, "I/O error occured" );
				return -1;
			}
		}
	}

	return 0;
}


// SOCK_ERROR
int Communication::send( const char *buf, int len ) {
	int ret;
	char nul = END_CHAR;

	ret = write( buf, len );
	if ( ret != -1 )
		ret = write( &nul, 1 );

	if ( ret != -1 )
		ret = flush();

	return ret;
}


// SOCK_ERROR
int Communication::finish() {
	int ret;
	char nul = END_CHAR;

	ret = write( &nul, 1 );

	if ( ret != -1 )
		ret = flush();

	return ret;
}

// len cannot be larger than the buffer size
// returns bytes read
// SOCK_ERROR, SOCK_CLOSED
int Communication::read( char *buf, int len ) {
	int ret = 0;

	if ( len < 1 ) return 0;

	if ( r_comm ) {

		ret = r_comm->read( buf, len );
		if ( ret < 0 ) {

			copy_err( r_comm );
			return -1;
		}

#ifdef _DEBUG
	fprintf(stdout, "Communication::read() buf '%s'\n", buf );
#endif

	}
	else {
		// initialize the time struct, as Linux modifies it every time
		// select() gets called
		read_arg.timeout.tv_sec  = timeout.tv_sec;
		read_arg.timeout.tv_usec = timeout.tv_usec;

		if ( use_custom_io )
			ret = buf_read( &buff_r, buf, len, (void *)this );
		else
			// ret = bytes read, SOCK_WOULDBLOCK, SOCK_ERROR
			ret = buf_read( &buff_r, buf, len, (void *)&read_arg );
		if ( ret < 0 ) {
			if ( ret == SOCK_ERROR && sock_err ) {

				Comm_ERROR( SOCK_ERROR, "Read Socket: got EPIPE" );
				return -1;
			}
			else if ( ret == SOCK_CLOSED && sock_err ) {

				Comm_ERROR( SOCK_CLOSED,
					"Read Socket: FIN received on socket" );
				return -1;
			}
			else if ( ret == COMM_TIMEOUT ) {

				Comm_ERROR( COMM_TIMEOUT,
					"Read Socket: connection timed out" );
				return -1;
			}
			else if ( ret == COMM_SELECTFAILED ) {

				Comm_ERROR( COMM_SELECTFAILED, strerror( errno ) );
				return -1;
			}
			else if ( ret == SOCK_WOULDBLOCK && sock_err ) {

				Comm_TRACE( SOCK_WOULDBLOCK,
					"Read Socket: WOULD BLOCK" );
				return -1;
			}
			else if ( use_custom_io ) {

				Comm_ERROR( ret, "Custom I/O error" );
				return -1;
			}
			else {

				Comm_ERROR( COMM_IO_ERROR, "I/O error occured" );
				return -1;
			}
		}
	}

	return ret;
}


// no problems if part of the data consists of control or null characters
int Communication::readline( char *buf, int len, char end_char ) {
	int ret = 0;
	char ch;
	int bytes_ret = 0;

	if ( len < 2 ) return 0;   // sanity check

	if ( r_comm ) {

		ret = r_comm->readline( buf, len, end_char );
		if ( ret < 0 ) {

			copy_err( r_comm );
			return -1;
		}
	}
	else {
		do {
			// initialize the time struct, as Linux modifies it every time
			// select() gets called
			read_arg.timeout.tv_sec  = timeout.tv_sec;
			read_arg.timeout.tv_usec = timeout.tv_usec;

			if ( use_custom_io )
				ret = buf_read( &buff_r, &ch, 1, (void *)this );
			else
				// ret = bytes read, SOCK_WOULDBLOCK, SOCK_ERROR
				ret = buf_read( &buff_r, &ch, 1, (void *)&read_arg );
			if ( ret < 0 ) {
				if ( ret == SOCK_ERROR && sock_err ) {

					Comm_ERROR( SOCK_ERROR,
						"Read Socket: got EPIPE" );
					return -1;
				}
				else if ( ret == SOCK_CLOSED && sock_err ) {

					Comm_ERROR( SOCK_CLOSED,
						"Read Socket: FIN received on socket" );
					return -1;
				}
				else if ( ret == COMM_TIMEOUT ) {

					Comm_ERROR( COMM_TIMEOUT,
						"Read Socket: connection timed out" );
					return -1;
				}
				else if ( ret == COMM_SELECTFAILED ) {

					Comm_ERROR( COMM_SELECTFAILED,
						strerror( errno ) );
					return -1;
				}
				else if ( ret == SOCK_WOULDBLOCK && sock_err ) {

					Comm_TRACE( SOCK_WOULDBLOCK,
						"Read Socket: WOULD BLOCK" );
					return -1;
				}
				else if ( use_custom_io ) {

					Comm_ERROR( ret, "Custom I/O error" );
					return -1;
				}
				else {

					Comm_ERROR( COMM_IO_ERROR, "I/O error occured" );
					return -1;
				}
			}
			else {
				// everything OK, so append the byte to buf
				buf[bytes_ret++] = ch;
			}
		} while ( ch != end_char && bytes_ret < len - 1 );

		buf[bytes_ret++] = '\0';

		return bytes_ret;
	}

	return ret;
}


int Communication::read_spacesplit(
		string arg_list[],
		int arg_list_len,
		int max_len,
		int end_char )
{
	int ret, count;
	char *arg_start, *arg_end;

	if ( max_len > bufsize ) {

		Comm_ERROR( COMM_ARG2LONG,
			"max_len argument in read_spacesplit() is "
			"longer than specified buffer size"
		);
		return -1;
	}

	ret = readline( read_buf, max_len, end_char );
	if ( ret == -1 || ret == 0 ) {

		// ret should never equal 0 anyway
		return -1;
	}
	else if ( ret == 1 ) {

		// this condition should not happen as well
		return 0;
	}

	// checking if the line received was ended properly
	if ( read_buf[ret - 2] != end_char ) {

		Comm_ERROR( COMM_BADCOMMAND, "Message did not terminate with a "
			"proper end character after maximum length reached"
		);
		return -1;
	}

	// terminate the string
	read_buf[ret - 2] = '\0';

	// split the input line in separate strings, delimited by
	// one or more spaces
	count = 0;
	arg_start = next_no_space( read_buf );
	while ( arg_start ) {
		arg_end = strchr( arg_start, ' ' );
		if ( arg_end ) {

			*arg_end = '\0';
			arg_list[count++] = arg_start;			
			if ( count == arg_list_len )
				break;
			arg_start = next_no_space( arg_end + 1 );
		}
		else
			break;
	}
	if ( arg_start && count < arg_list_len )
		arg_list[count++] = arg_start;			

	return count;
}

/* --------------------------------------------------------------------- */

void Communication::init( int timeout_sec ) {

	wbuffer  = new char[bufsize+1];
	rbuffer  = new char[bufsize+1];
	read_buf = new char[bufsize+1];

	// initialize the sets used by select()
	FD_ZERO( &write_arg.set );
	FD_ZERO( &read_arg.set );

	// initialize the time structure used by select()
	timeout.tv_sec = timeout_sec;
	timeout.tv_usec = 0;

	FD_ZERO( &write_custom_fd_set );
	FD_ZERO( &read_custom_fd_set );

	sock_err = false;

	r_comm = w_comm = 0;

	total_sent = 0L;
	total_compress_sent = 0L;
	total_recv = 0L;
	total_compress_recv = 0L;

}

/*
 * Returns pointer to next char after what ptr points to which is space.
 * If space is not found until '\0' is encountered, NULL is returned.
 */
char *Communication::next_no_space( char *ptr ) {

	while( *ptr == ' ' && *ptr != '\0' )
		ptr++;

	if ( *ptr == '\0' )
		return NULL;
	else
		return ptr;
}

void Communication::copy_err( Communication *comm ) {

	errstr = comm->errstr;
	errnum = comm->errnum;
}

