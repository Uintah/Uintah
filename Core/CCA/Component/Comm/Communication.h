
/* NOTE: Non-blocking I/O has not been tested at all! */

#ifndef _Communication_H_
#define _Communication_H_

#include <unistd.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/types.h>
#include <string.h>
#include <string>
#include "sock.h"
#include "buf.h"

// size of the input and the output buffer (two separate buffers are created)
#define DEFAULT_COMM_BUFSIZE 12000

// end character used by send() and finish() appended to designate end of line
#define END_CHAR '\n'

#define COMM_ARG2LONG -101
#define COMM_BADCOMMAND -102
#define COMM_BUFOVERFLOW -103
#define COMM_TIMEOUT -104
#define COMM_SELECTFAILED -105
#define COMM_IO_ERROR -106
#define COMM_EOF -107


/* ?????????????
 * add COMM_EOF support to the class
 */

#ifdef _INFO
#define Comm_ERROR(code,str) \
	{ \
		errnum = code; \
		if ( code > -1000 ) \
			errstr  = str; \
		printf( "Comm ERROR: %s\n", str ); \
	}
#else
#define Comm_ERROR(code,str) \
	do { \
	} while(0)
#endif

#ifdef _INFO
#define Comm_TRACE(code,str) \
	{ \
		errno = code; \
		if ( code > -1000 ) \
			errstr = str; \
		printf( "Comm TRACE: %s\n", str ); \
	}
#else
#define Comm_TRACE(code,str) \
	do { \
	} while(0)
#endif

class Communication;

/* A pointer to this struct is being used for the buffered reads/writes
 * while calling buf_XXX() functions.  See buf.h for more info.
 */
struct sock_arg {

	fd_set set;
	struct timeval timeout;
};


int write_abstract( int fd, char *buf, int len, void *arg );
int read_abstract( int fd, char *buf, int len, void *arg );
int write_abstract_custom( int fd, char *buf, int len, void *arg );
int read_abstract_custom( int fd, char *buf, int len, void *arg );

class Communication {
public:
	/* Communication()
	 * Constructs a Communications object out of an initialized and
	 * connected file descriptor.  Opening and closing connections is
	 * not a business of this class.  Hence, the file descriptor must be
	 * closed after it is used by the caller.  It can use both blocking
	 * and non-blocking I/O with its operations.  The reads and writes
	 * are buffered to improve on efficiancy.  You can also set a timeout
	 * on how much to wait at most for input, or for the system to become
	 * available for output.
	 * If socket error handling is not used, you can use error checking
	 * same as for system's read() and write() operations.
	 * NOTE: Non-blocking I/O has not been tested at all!
	 * Parameters:
	 *   fd -- a file descriptor which is bound to a connection
	 *   timeout_sec -- a maximum time to wait for the system for input
	 *                  or output; in seconds.  Leave it to zero if
	 *                  you want infinite wait time.  For non-blocking
	 *                  I/O, leave it to zero.
	 *   buffer_size -- size of the buffer -- must be 1 or more
	 */
	explicit Communication( int fd, int timeout_sec = 0,
		int buffer_size = DEFAULT_COMM_BUFSIZE )
		: fd( fd ), wfd( -1 ), 
                  bufsize(buffer_size), use_custom_io(false),
		  read_fn(0), write_fn(0),
		  w_fn_arg(0), r_fn_arg(0)
	{
		init( timeout_sec );

		// initialize the buffered I/O
		buf_init( &buff_w, fd, wbuffer, bufsize, write_abstract );
		buf_init( &buff_r, fd, rbuffer, bufsize, read_abstract );
	}

	/* Communication()
	 * Creates Comm. object using different file desc. for reading and
	 * writing, with given timeout and buffer size.
	 */
	Communication( int rfd, int wfd, int timeout_sec, int buffer_size )
		: fd( rfd ), wfd( wfd ), 
                  bufsize(buffer_size), use_custom_io(false),
		  read_fn(0), write_fn(0),
		  w_fn_arg(0), r_fn_arg(0)
	{
		init( timeout_sec );

		// initialize the buffered I/O
		buf_init( &buff_w, wfd, wbuffer, bufsize, write_abstract );
		buf_init( &buff_r, fd, rbuffer, bufsize, read_abstract );
	}

	/* Communication()
	 * Variation of the constructor above with input piped from
	 * another Communication class.
	 */
	Communication( Communication *r_comm, int wfd, int timeout_sec,
			int buffer_size )
		: fd( -1 ), wfd( wfd ), 
                  bufsize(buffer_size), use_custom_io(false),
		  read_fn(0), write_fn(0),
		  w_fn_arg(0), r_fn_arg(0)
	{
		init( timeout_sec );

		// initialize the buffered I/O
		buf_init( &buff_w, wfd, wbuffer, bufsize, write_abstract );

		this->r_comm = r_comm;
	}

	/* Communication()
	 * Variation of the constructor above with output piped from
	 * another Communication class.
	 */
	Communication( int rfd, Communication *w_comm, int timeout_sec,
			int buffer_size )
		: fd( rfd ), wfd( -1 ), 
                  bufsize(buffer_size), use_custom_io(false),
		  read_fn(0), write_fn(0),
		  w_fn_arg(0), r_fn_arg(0)
	{
		init( timeout_sec );

		// initialize the buffered I/O
		buf_init( &buff_r, fd, rbuffer, bufsize, read_abstract );

		this->w_comm = w_comm;
	}

	/* Communication()
	 * Variation of the constructor above with input and output piped from
	 * another Communication class.
	 */
	Communication( Communication *r_comm, Communication *w_comm,
			int timeout_sec, int buffer_size )
		: fd( -1 ), wfd( -1 ), 
                  bufsize(buffer_size), use_custom_io(false),
		  read_fn(0), write_fn(0),
		  w_fn_arg(0), r_fn_arg(0)
	{
		init( timeout_sec );

		this->r_comm = r_comm;
		this->w_comm = w_comm;
	}

	/* Communication()
	 * Creates Comm. object with reading and writing file descriptors.
	 * Here you can provide custom read and write functions which will be
	 * called whenever a block of data needs to be written or a read is
	 * needed.  w_fn_arg, r_fn_arg are arguments which will be passed when
	 * the custom functions are called.  comm is a pointer to the current
	 * Communication object.
	 * On success, the write function should return 0; the read function
	 * should return number of bytes read.  On error, they should return
	 * error codes of less than or equal to -1000.  The comm object passed
	 * along may be used to to call set_errmsg() to set a custom error
	 * message.  Your function does not need to set the error code, as it
	 * will be set automatically from its return value.  To read the error
	 * code and message call errmsg() and errcode() after the Comm. method
	 * failed.  If you don't need a distiction with various error code,
	 * just make it return COMM_IO_ERROR.  Optionally, return COMM_EOF
	 * if intput stream reached EOF.
	 * If timeout is set to a value other than zero, a select() will be
	 * called on the given file descriptor.  If descriptor is not available
	 * in timely fashion, errcode() will return COMM_TIMEOUT on the called
	 * method.  errcode() will return COMM_SELECTFAILED if select() fails.
	 * select() is not used at all if timeout is 0.
	 */
	Communication( int rfd, int wfd,
		int (*write_fn)(int, char *, int, Communication *comm, void *),
		int (*read_fn)(int, char *, int, Communication *comm, void *),
		void *w_fn_arg, void *r_fn_arg,
		int timeout_sec = 0,
		int buffer_size = DEFAULT_COMM_BUFSIZE )
			: fd( rfd ), wfd( wfd ),
                        bufsize(buffer_size), use_custom_io(true),
			read_fn(read_fn), write_fn(write_fn),
			w_fn_arg(w_fn_arg), r_fn_arg(r_fn_arg)
	{
		init( timeout_sec );

		// initialize the buffered I/O
		buf_init( &buff_w, wfd, wbuffer, bufsize,
			write_abstract_custom );
		buf_init( &buff_r, fd, rbuffer, bufsize,
			read_abstract_custom );
	}

	/* Communication()
	 * Variation of the constructor above with input piped from
	 * another Communication class.
	 */
	Communication( Communication *r_comm, int wfd,
		int (*write_fn)(int, char *, int, Communication *comm, void *),
		int (*read_fn)(int, char *, int, Communication *comm, void *),
		void *w_fn_arg, void *r_fn_arg,
		int timeout_sec = 0,
		int buffer_size = DEFAULT_COMM_BUFSIZE )
			: fd( -1 ), wfd( wfd ),
                        bufsize(buffer_size), use_custom_io(true),
			read_fn(read_fn), write_fn(write_fn),
			w_fn_arg(w_fn_arg), r_fn_arg(r_fn_arg)
	{
		init( timeout_sec );

		// initialize the buffered I/O
		buf_init( &buff_w, wfd, wbuffer, bufsize,
			write_abstract_custom );

		this->r_comm = r_comm;
	}

	/* Communication()
	 * Variation of the constructor above with output piped from
	 * another Communication class.
	 */
	Communication( int rfd, Communication *w_comm,
		int (*write_fn)(int, char *, int, Communication *comm, void *),
		int (*read_fn)(int, char *, int, Communication *comm, void *),
		void *w_fn_arg, void *r_fn_arg,
		int timeout_sec = 0,
		int buffer_size = DEFAULT_COMM_BUFSIZE )
			: fd( rfd ), wfd( -1 ),
                        bufsize(buffer_size), use_custom_io(true),
			read_fn(read_fn), write_fn(write_fn),
			w_fn_arg(w_fn_arg), r_fn_arg(r_fn_arg)
	{
		init( timeout_sec );

		// initialize the buffered I/O
		buf_init( &buff_r, fd, rbuffer, bufsize,
			read_abstract_custom );

		this->w_comm = w_comm;
	}

	/* Communication()
	 * Variation of the constructor above with input and output piped from
	 * another Communication class.
	 */
	Communication( Communication *r_comm, Communication *w_comm,
		int (*write_fn)(int, char *, int, Communication *comm, void *),
		int (*read_fn)(int, char *, int, Communication *comm, void *),
		void *w_fn_arg, void *r_fn_arg,
		int timeout_sec = 0,
		int buffer_size = DEFAULT_COMM_BUFSIZE )
			: fd( -1 ), wfd( -1 ),
                        bufsize(buffer_size), use_custom_io(true),
			read_fn(read_fn), write_fn(write_fn),
			w_fn_arg(w_fn_arg), r_fn_arg(r_fn_arg)
	{
		init( timeout_sec );

		this->r_comm = r_comm;
		this->w_comm = w_comm;
	}

	~Communication() {

		if ( wbuffer )
			delete [] wbuffer;

		if ( rbuffer )
			delete [] rbuffer;

		if ( read_buf )
			delete [] read_buf;

	}

	/* socketerr()
	 * When first constructor is used for reading from a socket connection,
	 * enables/disables socket-friendly error codes.  By default, it is
	 * disabled.  Only works if the first constructor is used.
	 */
	void socketerr( bool set ) { sock_err = (use_custom_io ? false : set); }

	/* supports_compression()
	 * Returns true if this implementation supports compression
	 */
	bool supports_compression() {
		return false;
	}

	/* Input methods */

	/* read()
	 * Reads data.
	 * Parameters:
	 *   buf - buffer to store input data to
	 *   len - length of buffer
	 * Returns:
	 *   Number of bytes received, but never zero; or
	 *   -1 if error occured. Use errmsg() to get the error msg, or
	 *      errcode() to get the error code.  Reasons for failure may be:
	 *        If socket error handling is used:
	 *          COMM_TIMEOUT -- if waiting for available read timed out
	 *          COMM_SELECTFAILED -- if select() failed; check errno
	 *          SOCK_WOULDBLOCK -- if non-blocking I/O used when read()
	 *                             would block
	 *          SOCK_ERROR -- if writing to the socked failed
	 *          SOCK_CLOSED -- if reading from a closed connection
	 *        If socket error handling is NOT used:
	 *          COMM_TIMEOUT -- if waiting for available read timed out
	 *          COMM_SELECTFAILED -- if select() failed; check errno
	 *          COMM_ERROR_IO - if I/O failed for any other reason.  You may
	 *            be able to check (for example) errno for more info.
	 *        If custom read/write function are used:
	 *          COMM_TIMEOUT -- if waiting for available read timed out
	 *          COMM_SELECTFAILED -- if select() failed; check errno
	 *          COMM_ERROR_IO or any value less than or equal to -1000, as
	 *            returned from the custom read function.
	 */
	int read( char *buf, int len );

	/* readline()
	 * Will return a string from the input stream until end_char
	 * (designating end of line) is found.  The end_char is also stored in
	 * the input, and '\0' is put to end the string.  To check if
	 * end_char was found and the line was not too long, user must use the
	 * return value to check: if buf[retvalue-2] == end_char, then
	 * it the line was read a whole.
	 * Returns:
	 *   See read(). Only difference is that the return value here will
	 *   never equal 0 or 1, plus the return value would include the '\0'
	 *   put at the end.
	 */
	int readline( char *buf, int len, char end_char );

	/* read_spacesplit()
	 * Reads up to buffer size long line (Line being defined as a sequence
	 * of characters until (and including) end_len is found), and returns
	 * an array of strings with all of the words found.  A word is a
	 * sequence of charaters delimited by  one or more spaces, or by
	 * end_char.
	 * Parameters:
	 *   arg_list - pointer to a string array
	 *   arg_list_len - size of the array
	 *   max_len - maximum size of the array, not being greater than
	 *             the buffer size
	 *   end_char - the char to designate an end of a line
	 * Returns:
	 *   number of arguments found and stored in the string array;
	 *   0 if no valid arguments were found;
	 *   -1 if error occured. Use errmsg() to get the error msg, or
	 *      errcode() to get the error code.  Reasons for failure may be:
	 *        If socket error handling is used:
	 *          COMM_TIMEOUT -- if waiting for available read timed out
	 *          COMM_SELECTFAILED -- if select() failed; check errno
	 *          SOCK_WOULDBLOCK -- if non-blocking I/O used when read()
	 *                           would block
	 *          SOCK_ERROR - socket input error
	 *          SOCK_CLOSED - socket was closed by peer
	 *          COMM_ARG2LONG - the max_len value exceeded the buffer size
	 *          COMM_BADCOMMAND - Message did not terminate with a
	 *            proper end character after maximum length reached
	 *        If socket error handling is NOT used:
	 *          COMM_TIMEOUT -- if waiting for available read timed out
	 *          COMM_SELECTFAILED -- if select() failed; check errno
	 *          COMM_ARG2LONG - the max_len value exceeded the buffer size
	 *          COMM_BADCOMMAND - Message did not terminate with a
	 *          COMM_ERROR_IO - if I/O failed for any other reason.  You may
	 *            be able to check (for example) errno for more info.
	 *        If custom read/write function are used:
	 *          COMM_TIMEOUT -- if waiting for available read timed out
	 *          COMM_SELECTFAILED -- if select() failed; check errno
	 *          COMM_ARG2LONG - the max_len value exceeded the buffer size
	 *          COMM_BADCOMMAND - Message did not terminate with a
	 *          COMM_ERROR_IO or any value less than or equal to -1000, as
	 *            returned from the custom read function.
	 */
	int read_spacesplit(
		std::string arg_list[],
		int arg_list_len,
		int max_len,
		int end_char );

	/* Output methods */

	/* write()
	 * Sends a buffer of data to the socket, no END_CHAR character
	 * is added at the end.
	 * Parameters:
	 *   buf - the buffer of data to send
	 *   len - the length of the buffer
	 * Returns:
	 *   0 - sucess;
	 *   -1 if error occured. Use errmsg() to get the error msg, or
	 *      errcode() to get the error code.  Reasons for failure may be:
	 *        If socket error handling is used:
	 *          COMM_TIMEOUT -- if waiting for available write timed out
	 *          COMM_SELECTFAILED -- if select() failed; check errno
	 *          SOCK_WOULDBLOCK -- if non-blocking I/O used when write()
	 *                             would block
	 *          SOCK_ERROR -- if writing to the socked failed
	 *        If socket error handling is NOT used:
	 *          COMM_TIMEOUT -- if waiting for available write timed out
	 *          COMM_SELECTFAILED -- if select() failed; check errno
	 *          COMM_ERROR_IO - if I/O failed for any other reason.  You may
	 *            be able to check (for example) errno for more info.
	 *        If custom read/write function are used:
	 *          COMM_TIMEOUT -- if waiting for available write timed out
	 *          COMM_SELECTFAILED -- if select() failed; check errno
	 *          COMM_ERROR_IO or any value less than or equal to -1000, as
	 *            returned from the custom write function.
	 */
	int write( const char *buf, int len );

	/* send()
	 * Sends a buffer of data to the socket, appends END_CHAR at the end,
	 * and flushes the buffer using flush().
	 * Parameters:
	 *   buf - the buffer of data to send
	 *   len - the length of the buffer
	 * Returns:
	 *   See write().
	 */
	int send( const char *buf, int len );

	/* flush()
	 * Writes the contents of the buffer to the socket connection.
	 * Returns:
	 *   See write().
	 */
	int flush();

	/* finish()
	 * Sends a END_CHAR character to the socket and flushes the buffer.
	 * Parameters:
	 *   none.
	 * Returns:
	 *   See write().
	 */
	int finish();

	/* Error handling methods */

	/* set_errmsg()
	 * When custom functions are used, you may set a custom error message.
	 */
	void set_errmsg( const std::string& str ) { errstr = str; }

	/* set_errcode()
	 * When custom functions are used, you may set a custom error code.
	 */
	void set_errcode( int num ) { errnum = num; }

	/* errmsg()
	 * Returns a string of the error message.  This command should be
	 * called only after an error occured.
	 */
	inline std::string& errmsg() { return errstr; }

	/* errcode()
	 * Returns error code, which may one of these values:
	 * COMM_ARG2LONG, COMM_BUFOVERFLOW, SOCK_ERROR, SOCK_CLOSED,
	 * COMM_BADCOMMAND.  This command should be called only when error
	 * occured.
	 */
	inline int errcode() { return errnum; }

private:
	void init( int timeout_sec );
	void copy_err( Communication *comm );

	char *next_no_space( char *ptr );
	int fd;  /* for socket file desc. this is for both reading and writing;
	          * if this class is initialized with 2 desc., this is read */
	int wfd; /* if this class is init. with 2 desc., this is write */
	         /* watch, it's used in flush_end_compression() */

	char *read_buf;

	// error string and error code
	std::string errstr;
	int errnum;

	// used with the buffered I/O operations
	bufferer buff_w;
	bufferer buff_r;
	char *wbuffer;
	char *rbuffer;

	// select utilities
	sock_arg read_arg, write_arg;

	int bufsize;  // size of the buffer - always positive; 1 non-buffered

	bool use_custom_io;  // true if user custom I/O functions
	bool sock_err;       // true when socket-friendly error codes enabled
public:
	int (*read_fn)( int, char *, int, Communication *, void * );
	int (*write_fn)( int, char *, int, Communication *, void * );
	void *w_fn_arg;  // void * argument to write_fn
	void *r_fn_arg;  // void * argument to read_fn
	struct timeval timeout;
	fd_set read_custom_fd_set;
	fd_set write_custom_fd_set;

	// not NULL if input and/or output is piped from another Comm. class
	Communication *r_comm;
	Communication *w_comm;

	/* these may directly be used by other modules (grep) */
	long total_sent;
	long total_compress_sent; /* compression not used, this is added
	                           * up the same as total_sent */
	long total_recv;
	long total_compress_recv; /* compression not used, this is added
	                           * up the same as total_recv */

};

#endif
