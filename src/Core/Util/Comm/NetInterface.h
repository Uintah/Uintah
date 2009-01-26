/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


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
 * HEADER (H) FILE : NetInterface.h
 *
 * DESCRIPTION     : The NetInterface object is an abstracted interface to 
 *                   socket code.  It provides the ability to make and manage
 *                   socket connections and send/receive data on these 
 *                   connections.  This is a wrapper class for the C 
 *                   struct and functions in NetInterfaceC.h[c].
 *                     
 * AUTHOR(S)       : Jenny Simpson
 *                   SCI Institute
 *                   University of Utah
 *                 
 * CREATED         : Mon Apr 12 13:28:20 MDT 2004
 * MODIFIED        : Mon Apr 12 13:28:20 MDT 2004
 * DOCUMENTATION   :
 * NOTES           : 
 *
 * Copyright (C) 2003 SCI Group
*/

#ifndef SCI_Core_Util_NetInterface_h
#define SCI_Core_Util_NetInterface_h

#include <sci_defs/scisock_defs.h>

#ifdef HAVE_SCISOCK

// SCIRun includes
#include <Core/Malloc/Allocator.h>
//#include <SCISockInterface/lib/NetInterfaceC.h>
#include <Core/Util/Comm/NetConnection.h>
#include <NetInterfaceC.h>
//#include <Packages/DDDAS/Core/Utils/NetConnection.h>

// Standard lib includes
#include <iostream>
#include <cassert>
#include <limits>
#include <vector>

#define MAX_PORT_LEN 8

namespace SCIRun {

// ****************************************************************************
// ************************** Class: NetInterface *****************************
// ****************************************************************************

class NetInterface 
{

public:

  //! Constructors
  NetInterface();

  //! Copy constructor
  NetInterface( const NetInterface& nl );

  //! Destructor
  ~NetInterface();

  //! Utility functions

  // 
  // General functions
  //

  int stop(); 

  //
  // Listening functions
  // 

  void listen( int listen_port, int blocking, 
               void (*listen_callback)(NetInterface * interface, 
                                       NetConnection * conn) );
  int wait_listener();
  void stop_listen(); 

  //
  // Connection functions 
  //
 
  NetConnection connect( std::string host_name, int port );
  int disconnect( NetConnection * conn ); 
  int disconnect_all(); 
  void print_connections(); 
  std::vector<NetConnection> get_connections();

  // 
  // Reading functions
  //
  int start_read_loop( NetConnection * conn, 
                       void (*read_callback)( NetInterface * interface, 
                       NetConnection * conn, char * recv_buffer, 
                       int num_read) );
  void stop_read_loop( NetConnection * conn ); 
  int wait_reader( NetConnection * conn );
  int read( NetConnection * conn, char * recv_buffer, int max_len );
  int readn( NetConnection * conn, char * recv_buffer, int num_bytes );
  int readline( NetConnection * conn, char * recv_buffer, int max_len );
  int recv( NetConnection * conn, char * recv_buffer, int max_len, int flags );

  //
  // Writing functions
  //

  int writen( NetConnection * conn, char * buffer, int buffer_len );
  int write( NetConnection * conn, char * buffer, int num_bytes );
  int send( NetConnection * conn, char * buffer, int num_bytes, int flags );
  int send_message( NetConnection * conn, int message );

  //! Public member variables (treat as if they're private)

  // User specified callback to process new connections accepted by the 
  // listener
  void ( *listen_callback_ )( NetInterface * interface, 
                              NetConnection * conn ); 
  // User specified callback to process data read from a connection
  void ( *read_callback_ )( NetInterface * interface, NetConnection * conn, 
                            char * recv_buffer, int num_read );

private:

  //! Member variables
  
  // Low-level C object
  struct NetInterfaceC interface_;

};

} // End namespace SCIRun
 
#endif // HAVE_SCISOCK
#endif // SCI_Core_Util_NetInterface_h



