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
 * C++ (CC) FILE : NetConnection.cc
 *
 * DESCRIPTION   : The NetConnection object contains all information 
 *                 associated with a single socket connection.
 *                 This is a wrapper class for the C struct in 
 *                 NetConnectionC.h[c].  This functions as a "READ ONLY" 
 *                 object in that the object cannot be modified directly.
 *                 The only way to modify a NetConnection object is via 
 *                 the NetInterface object.
 *      
 * AUTHOR(S)     : Jenny Simpson
 *                 SCI Institute
 *                 University of Utah
 *                 
 * CREATED       : Mon Apr 12 13:39:33 MDT 2004
 * MODIFIED      : Wed May 26 11:57:46 MDT 2004
 * DOCUMENTATION :
 * NOTES         : 
 *
 * Copyright (C) 2003 SCI Group
*/
  
// SCIRun includes
#include <Core/Util/Comm/NetConnection.h>

#ifdef HAVE_SCISOCK

using namespace std;

namespace SCIRun {

/*===========================================================================*/
// 
// NetConnection
//
// Description : Constructor.  
//
// Arguments   : none
//
NetConnection::NetConnection()
{
  //cout << "(NetConnection::NetConnection) I'm alive!" << endl;
  connection_ = 0;
}

/*===========================================================================*/
// 
// NetConnection
//
// Description : Constructor.  
//
// Arguments   : 
//
// struct NetConnection * conn - Low-level NetConnectionC to wrap.
//
NetConnection::NetConnection( struct NetConnectionC * conn )
{
  if( conn == 0 )
  {
    cerr << "(NetConnection::NetConnection) WARNING: Connection is NULL" 
         << endl;
    connection_ = 0;
    return;
  }

  // Increment the reference count
  inc_ref( &(conn->ref_count_), &(conn->ref_count_mutex_) );
  connection_ = conn;
}

/*===========================================================================*/
// 
// NetConnection
//
// Description : Copy Constructor. This is only a shallow copy by design.
//
// Arguments   : 
//
// const NetConnection& nl - Instance of NetConnection to be copied.
//
NetConnection::NetConnection( const NetConnection& nl )
{
  // Increment the reference count
  inc_ref( &(nl.connection_->ref_count_), 
           &(nl.connection_->ref_count_mutex_) );
  connection_ = nl.connection_;
}

/*===========================================================================*/
// 
// ~NetConnection
//
// Description : Destructor
//
// Arguments   : none
//
NetConnection::~NetConnection()
{
  if( connection_ == 0 ) return;

  struct NetConnectionC * conn = connection_;

  // Decrement the reference count for this connection
  dec_ref( &(conn->ref_count_), &(conn->ref_count_mutex_), 
           net_connection_call_dest, (void *) conn );
}

/*===========================================================================*/
// 
// is_open
//
// Description : Returns true if the connection is open, false otherwise.
//
// Arguments   : none
//
bool NetConnection::is_open()
{
  if( connection_ == 0 ) return false;

  int open = net_connection_is_open( connection_ );

  bool ret = false;

  if( open == 1 ) ret = true;

  return ret;
}

/*===========================================================================*/
// 
// get_type
//
// Description : Returns the type of connection (i.e. PTP or MULTICAST).  
//               Types are defined in NetConnectionC.h.  Multicast connections
//               are not currently implemented.  Returns -1 if the type is
//               undefined.
//
// Arguments   : none
//
int NetConnection::get_type()
{
  if( connection_ == 0 ) return -1;

  return net_connection_get_type( connection_ );
}

/*===========================================================================*/
// 
// get_sockfd
//
// Description : Returns the socket descriptor for this connection.  Returns
//               -1 if there is no open socket for this connection.
//
// Arguments   : none
//
int NetConnection::get_sockfd()
{
  if( connection_ == 0 ) return -1;

  return net_connection_get_sockfd( connection_ );
}

/*===========================================================================*/
// 
// get_host_name
//
// Description : Returns the name of the remote host for this connection.
//               Returns an empty string if this connection is closed.
//
// Arguments   : none
//
string NetConnection::get_host_name()
{
  if( connection_ == 0 ) return "";

  char * host_name = net_connection_get_host_name( connection_ );
  if( host_name != 0 ) 
  {
    // Return string version of host_name
    string host_name_str = host_name;
    return host_name_str;
  }
  else
  {
    return "";
  }
}

/*===========================================================================*/
// 
// get_port
//
// Description : Returns the port number for remote host.  Returns -1 if this
//               connection is closed.
//
// Arguments   : none
//
int NetConnection::get_port()
{
  if( connection_ == 0 ) return -1;

  return net_connection_get_port( connection_ );
}

/*===========================================================================*/
// 
// print
//
// Description : Returns the port number for remote host.  Returns -1 if this
//               connection is closed.
//
// Arguments   : none
//
void NetConnection::print()
{
  string type_str;
 
  switch( net_connection_get_type(connection_) )
  {
    case PTP :
      type_str = "PTP";
      break;
    case MULTICAST :
      type_str = "MULTICAST";
      break;
    default :
      type_str = "UNKNOWN";
  }

  string status = "NOT OPEN";

  if( is_open() ) status = "OPEN"; 

  cout << "(NetConnection::print) Connection Type : " << type_str << endl;
  cout << "(NetConnection::print) Connection Status : " << status << endl;
  cout << "(NetConnection::print) Socket Descriptor : " << get_sockfd() 
       << endl;
  cout << "(NetConnection::print) Remote Host : " << get_host_name() << endl;
  cout << "(NetConnection::print) Remote Port : " << get_port() << endl;

}

/*===========================================================================*/
// 
// get_connection
//
// Description : Returns the lower level connection object associated with this
//               connection.  This function should be private, but NetInterface
//               needs access to it so it's public.  PLEASE DO NOT USE THIS 
//               FUNCTION!
//
// Arguments   : none
//
struct NetConnectionC * NetConnection::get_connection()
{
  return connection_;
}
   
} // End namespace SCIRun

#endif
