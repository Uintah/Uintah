/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/
   
  
/*
 * C++ (CC) FILE : NetInterface.cc
 *
 * DESCRIPTION   : The NetInterface object is an abstracted interface to 
 *                 socket code.  It provides the ability to make and manage
 *                 socket connections and send/receive data on these 
 *                 connections.  This is a wrapper class for the C 
 *                 struct and functions in NetInterfaceC.h[c].
 *                       
 * AUTHOR(S)     : Jenny Simpson
 *                 SCI Institute
 *                 University of Utah
 *                 
 * CREATED       : Mon Apr 12 13:39:33 MDT 2004
 * MODIFIED      : Mon Apr 12 13:39:33 MDT 2004
 * DOCUMENTATION :
 * NOTES         : 
 *
 * Copyright (C) 2003 SCI Group
*/
    
// SCIRun includes
#include <Core/Util/Comm/NetInterface.h>

#ifdef HAVE_SCISOCK

 
using namespace std;

namespace SCIRun {
 
// Helper functions
void listen_callback_converter( struct NetConnectionC * conn, void * arg );
void read_callback_converter( struct NetConnectionC * conn, 
                              char * recv_buffer, int num_read, void * arg );

/*===========================================================================*/
// 
// NetInterface
//
// Description : Constructor
//
// Arguments   : none
//
NetInterface::NetInterface()
{
  //cout << "(NetInterface::NetInterface) I'm alive!" << endl;

  net_interface_init( &interface_ );

  listen_callback_ = 0;
  read_callback_ = 0;
}
 
/*===========================================================================*/
// 
// NetInterface
//
// Description : Copy Constructor.  I've done nothing because the user should
//               not have multiple copies of the same NetInterface floating 
//               around.  There should only be one copy at any given time.
//
// Arguments   : 
//
// const NetInterface& nl - Instance of NetInterface to be copied.
//
NetInterface::NetInterface( const NetInterface& nl )
{
  // Do nothing
} 

/*===========================================================================*/
// 
// NetInterface
//
// Description : Destructor
//
// Arguments   : none
//
NetInterface::~NetInterface()
{
  //cout << "(NetInterface::~NetInterface) Inside" << endl;
  net_interface_destroy( &interface_ );
}

/*===========================================================================*/
// 
// stop
//
// Description : Stop listening for incoming connection requests and shut down
//               all connections. 
//
// Arguments   : none
//
int NetInterface::stop()
{
  return net_interface_stop_all( &interface_ );
}

/*===========================================================================*/
// 
// listen
//
// Description : Listen for incoming connection requests as a blocking or 
//               non-blocking server.  Register a callback function to 
//               process new connections accepted by the listener.
//
// Arguments   : 
//
// int listen_port - Port to listen on.
//
// int blocking - Should be set to 0 or 1.  If 0, this is a non-blocking server
//                meaning that it is run on a new, separate thread. If 1, this 
//                is a blocking server meaning that it is run on the main 
//                thread.  If blocking is set to something other than 0 or 1,
//                the server is a blocking server.
//
// void (*listen_callback)(struct NetConnectionC * conn) - 
//    Callback function that is called whenever a new connection is accepted
//    by the listener.
//
void NetInterface::listen( int listen_port, int blocking, 
                           void (*listen_callback)( NetInterface * interface,
                                                    NetConnection * conn) )
{
  // Register user specified listen callback
  listen_callback_ = listen_callback;
  
  net_interface_listen( &interface_, listen_port, blocking, 
                        listen_callback_converter, (void *) this );
}

/*===========================================================================*/
// 
// wait_listener
//
// Description : Wait for the listening thread to die.  This is a blocking
//               call.  Returns the status of the dead thread, or -1 if the
//               listening thread id is NULL.
//
// Arguments   : none
//
int NetInterface::wait_listener()
{
  int status;
  status = net_interface_wait_listener( &interface_ );
  return status;
}

/*===========================================================================*/
// 
// stop_listen
//
// Description : Stop listening for new connection requests.
//
// Arguments   : none
//
void NetInterface::stop_listen()
{
  net_interface_stop_listen( &interface_ );
}

/*===========================================================================*/
// 
// connect
//
// Description : Opens a socket connection to the remote host.  Creates a
//               NetConnection object with the connected socket and adds the
//               object to the list of connections.  Returns a NetConnection
//               that is open on success, closed on failure.  Check the 
//               outcome using connection.is_open().  Always check to make 
//               sure a connection object is open before using it.
//
// Arguments   : 
//
// string host_name - String IP address or host name of remote host to connect
//                    to.
//
// int port - Listening port to connect to on remote host.
//
NetConnection NetInterface::connect( string host_name, int port )
{
  //cout << "(NetInterface::connect) Inside" << endl;

  struct NetConnectionC * real_conn;
  char * host_name_c = const_cast<char*>( host_name.c_str() );
  int host_name_len = host_name.size();

  real_conn = net_interface_connect( &interface_, host_name_c, host_name_len, 
                                     port );  

  //cout << "(NetInterface::connect) Returning" << endl;
  return NetConnection( real_conn );
}

/*===========================================================================*/
// 
// disconnect 
//
// Description : Disconnects the specified connection.  Removes this 
//               connection from the list of 
//               connections.  Returns 0 on success, -1 on failure.
//
// Arguments   : 
//
// NetConnection * conn - Connection to disconnect.
//
int NetInterface::disconnect( NetConnection * conn )
{
  struct NetConnectionC * real_conn = conn->get_connection();
  return net_interface_disconnect( &interface_, real_conn );  
}

/*===========================================================================*/
// 
// disconnect_all 
//
// Description : Disconnect from all connections.  Returns 0 on success, -1
//               on failure.
//
// Arguments   : none
//
int NetInterface::disconnect_all()
{
  return net_interface_disconnect_all( &interface_ );  
}


/*===========================================================================*/
// 
// print_connections
//
// Description : Print information (host name, port, and socket) for all 
//               open connections.
//
// Arguments   : none
//
void NetInterface::print_connections()
{
  net_interface_print_conns( &interface_ );

  /*
  // Get the connections
  std::vector<NetConnection> connections = get_connections();

  // Print each connection
  int num_conns = connections.size();

  cout << "(print_connections) There are " << num_conns << " connections"  
       << endl;

  for( int i = 0; i < num_conns; i++ )
  {
    (connections[i]).print();
  }
  */
}

/*===========================================================================*/
// 
// get_connections
//
// Description : Return a vector of all the connections.
//
// Arguments   : none
//
std::vector<NetConnection> NetInterface::get_connections()
{
  std::vector<NetConnection> connections;

  // Convert all the NetConnectionC objects in the connections array to 
  // NetConnection objects and add them to the array.
  for( int i = 0; i < interface_.num_connections_; i++ )
  {
    NetConnection conn( interface_.connections_[i] );
    connections.push_back( conn );    
  } 
  return connections;
}

/*===========================================================================*/
// 
// start_read_loop 
//
// Description : Continually read from a specified connection.  Returns 0 if
//               the read_loop is successfully started, -1 on error.
//
// Arguments   : 
//
// NetConnection * conn - Connection to read from.
//
// void (*read_callback)( NetConnection conn, char * recv_buffer, int num_read)
//     Callback function to handle data as it is read from the socket.
//
int NetInterface::start_read_loop( NetConnection * conn, 
                                   void (*read_callback)( 
                                   NetInterface * interface, 
                                   NetConnection * conn, 
                                   char * recv_buffer, int num_read) )
{
  // Register user specified read callback
  read_callback_ = read_callback;
  
  struct NetConnectionC * real_conn = conn->get_connection();
  return net_interface_read_loop( &interface_, real_conn, 
                                  read_callback_converter, (void *) this );
}

/*===========================================================================*/
// 
// stop_read_loop 
//
// Description : Stop the read thread for this conneciton.
//
// Arguments   : 
//
// NetConnection * conn - Connection whose read thread we want to 
//                        stop.
//
void NetInterface::stop_read_loop( NetConnection * conn )
{
  struct NetConnectionC * real_conn = conn->get_connection();
  net_interface_stop_read_loop( &interface_, real_conn );
}

/*===========================================================================*/
// 
// wait_reader
//
// Description : Wait for the reader thread for a particular socket connection 
//               to die.  This is a blocking call.  Returns the status of the 
//               dead thread, or -1 on error.
//
// Arguments   : 
//
// NetConnection * conn - Connection whose reader we are waiting for.
//
int NetInterface::wait_reader( NetConnection * conn )
{
  struct NetConnectionC * real_conn = conn->get_connection();
  return net_interface_wait_reader( &interface_, real_conn );
}

/*===========================================================================*/
// 
// read
//
// Description : Explicitly read data from the connection.  Wrapper for the 
//               socket read() function. Returns the number of bytes read, or 
//               -1 on failure.
//
// Arguments   : 
//
// NetConnection * conn - Connection to read from.
//
// char * recv_buffer - Buffer to hold data read from socket.
//
// int max_len - Maxmum number of bytes to be read and stored in the buffer.
//
int NetInterface::read( NetConnection * conn, char * recv_buffer, int max_len )
{
  struct NetConnectionC * real_conn = conn->get_connection();
  return net_interface_read( &interface_, real_conn, recv_buffer, max_len );
}

/*===========================================================================*/
// 
// readn
//
// Description : Explicitly read some number of bytes of data from the socket.
//               Wrapper for Steven's readn() function. Returns the number 
//               of bytes read, or -1 on error.
//
// Arguments   : 
//
// NetConnection * conn - Connection to read from.
//
// char * recv_buffer - Buffer to hold data read from socket.
//
// int num_bytes - Number of bytes to be read and stored in the buffer.
//
int NetInterface::readn( NetConnection * conn, char * recv_buffer, 
                         int num_bytes )
{
  struct NetConnectionC * real_conn = conn->get_connection();
  return net_interface_readn( &interface_, real_conn, recv_buffer, num_bytes );
}

/*===========================================================================*/
// 
// readline
//
// Description : Explicitly read data from the socket until one of these
//               conditions has been met:
//               - Read newline '\n' character
//               - Read EOF
//               - Read maxlen characters
//               - An error is encountered
//
//               Wrapper for Steven's readline() function. Returns the number 
//               of bytes read, or -1 on error.
//
// Arguments   : 
//
// NetConnection * conn - Connection to read from.
//
// char * recv_buffer - Buffer to hold data read from socket.
//
// int max_len - Maxmum number of bytes to be read and stored in the buffer.
//
int NetInterface::readline( NetConnection * conn, char * recv_buffer, 
                            int max_len )
{
  struct NetConnectionC * real_conn = conn->get_connection();
  return net_interface_readline( &interface_, real_conn, recv_buffer, 
                                 max_len );
}

/*===========================================================================*/
// 
// recv
//
// Description : Explicitly read data from the socket.  Wrapper for the socket
//               recv() function. Returns the number of bytes read, or -1 on 
//               failure.
//
// Arguments   : 
//
// NetConnection * conn - Connection to read from.
//
// char * recv_buffer - Buffer to hold data read from socket.
//
// int max_len - Maxmum number of bytes to be read and stored in the buffer.
//
// int flags - Either 0 or formed by logical OR'ing one or more constants.
// 
int NetInterface::recv( NetConnection * conn, char * recv_buffer, int max_len, 
                        int flags )
{
  struct NetConnectionC * real_conn = conn->get_connection();
  return net_interface_recv( &interface_, real_conn, recv_buffer, 
                             max_len, flags );
}

/*===========================================================================*/
// 
// writen
//
// Description : Write a specific number of bytes to this connection.  
//               Returns 0 if successful, -1 otherwise.
//
// Arguments   : 
//
// NetConnection * conn - Connection to write data to.
//
// char * buffer - Bytes to write to socket.
//
// int num_bytes - Number of bytes to write.
//
int NetInterface::writen( NetConnection * conn, char * buffer, int num_bytes )
{
  struct NetConnectionC * real_conn = conn->get_connection(); 
  
  return net_interface_writen( &interface_, real_conn, buffer, num_bytes );
}

/*===========================================================================*/
// 
// write
//
// Description : Write a specific number of bytes to this connection.  
//               Returns the number of bytes successfully written, -1 on error.
//               It is preferable to use net_interface_writen since this
//               keeps writing until all bytes are written.  write doesn't
//               gaurantee that all bytes will be written.
//
// Arguments   : 
//
// NetConnection * conn - Connection to write data to.
//
// char * buffer - Bytes to write to socket.
//
// int num_bytes - Length of buffer in bytes.
//
int NetInterface::write( NetConnection * conn, char * buffer, int num_bytes )
{
  struct NetConnectionC * real_conn = conn->get_connection(); 
  
  return net_interface_write( &interface_, real_conn, buffer, num_bytes );
}

/*===========================================================================*/
// 
// send
//
// Description : Similar to writen, but an addition argument (flags) can be
//               specified. Write a specific number of bytes to this 
//               connection.  Returns the number of bytes successfully written,
//               -1 on error.
//
// Arguments   : 
//
// NetConnection * conn - Connection to write data to.
//
// char * buffer - Bytes to write to socket.
//
// int num_bytes - Length of buffer in bytes.
//
// int flags - Either 0 or formed by logical OR'ing one or more constants.
//
int NetInterface::send( NetConnection * conn, char * buffer, int num_bytes, 
                        int flags )
{
  struct NetConnectionC * real_conn = conn->get_connection(); 
  
  return net_interface_send( &interface_, real_conn, buffer, num_bytes, 
                             flags );
}

/*===========================================================================*/
//  
// send_message
// 
// Description : Send one of a set of predefined messages to a connected
//               host.  These messages are defined in NetInterfaceC.h.
//               Returns 0 if successfull, -1 otherwise.
//
// Arguments   : 
//
// NetConnection * conn - Connection to send message to.
//
// int message_type - One of a set of predefined message types.
//
int NetInterface::send_message( NetConnection * conn, int message )
{
  struct NetConnectionC * real_conn = conn->get_connection(); 
  
  return net_interface_send_message( &interface_, real_conn, message );
}

/*===========================================================================*/
//  
// listen_callback_converter
// 
// Description : Callback function that processes each new connection
//               accepted by the listener.
//
// Arguments   : 
//
// struct NetConnectionC * conn - New connection to process.
//
// void * arg - Pointer to NetInterface object
//
void listen_callback_converter( struct NetConnectionC * conn, void * arg )
{
  //cout << "(listen_callback_converter) Inside" << endl;

  // Get arguments
  assert( arg != 0 );

  // Create a NetInterface object
  NetInterface * interface = (NetInterface *) arg;

  // Create NetConnection object
  NetConnection connection( conn );

  // Call user specified listen callback
  if( interface->listen_callback_ != 0 )
  {
    interface->listen_callback_( interface, &connection );
    //cout << "(listen_callback_converter) User callback returned" << endl;
  }
  else
  {
    cerr << "(listen_callback_converter) ERROR: "  
         << "No listening callback registered.  Aborting callback" << endl;
  }
  //cout << "(listen_callback_converter) Returning" << endl;
} 

/*===========================================================================*/
//  
// read_callback_converter
//
// Description : Callback function that processes data read from a connected
//               socket.
//
// Arguments   : 
//
// struct NetConnectionC * conn - Connection data was read from.
//
// char * recv_buffer - Buffer containing data read.
//
// int num_read - Number of bytes read from connection.
//
// void * arg - Pointer to NetInterface object.
//
void read_callback_converter( struct NetConnectionC * conn, 
                              char * recv_buffer, int num_read, void * arg )
{
  //cout << "(read_callback_converter) Inside" << endl;

  // Get arguments
  assert( arg != 0 );

  // Create a NetInterface object
  NetInterface * interface = (NetInterface *) arg;

  // Create NetConnection object
  NetConnection connection( conn );

  // Call user specified read callback
  if( interface->read_callback_ != 0 )
  {
    interface->read_callback_( interface, &connection, recv_buffer, num_read );
    //cout << "(read_callback_converter) User callback returned" << endl;
  }
  else
  {
    cerr << "(read_callback_converter) ERROR: "  
         << "No reading callback registered.  Aborting callback" << endl;
  }
  //cout << "(read_callback_converter) Returning" << endl;
} 

} // End namespace SCIRun

#endif
