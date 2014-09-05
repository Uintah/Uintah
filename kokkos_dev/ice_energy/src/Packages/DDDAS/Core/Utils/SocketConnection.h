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
 * HEADER (H) FILE : SocketConnection.h
 *
 * DESCRIPTION     : This represents the first, rough iteration of a wrapper
 *                   class for socket code in SCIRun.  The socket code from
 *                   the DDDAS StreamReader module has been added as a method
 *                   in this class.  No other infrastructure has been developed
 *                   for the class yet. 
 *                     
 * AUTHOR(S)       : Jenny Simpson
 *                   SCI Institute
 *                   University of Utah
 * 
 *                   Chad Shannon
 *          	     Center for Computational Sciences
 *                   University of Kentucky
 *                 
 * CREATED         : Mon Mar  1 16:33:34 MST 2004
 * MODIFIED        : Mon Mar  1 16:33:34 MST 2004
 * DOCUMENTATION   :
 * NOTES           : Much of the code contained in this class has been 
 *                   borrowed from the xccs package created by Chad Shannon. 
 *                   Most of the xccs code was stripped from XMMS-1.2.7 source 
 *                   code.  In any case, it is just basic socket code.
 *
 * Copyright (C) 2003 SCI Group
*/

#ifndef SocketConnection_h
#define SocketConnection_h

// SCIRun includes

// Standard lib includes

#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <assert.h>
#include <sys/types.h>

// Networking and C includes

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>

// XCCS defines

#define BUFFER_SIZE	4096
#define VERSION	"0.0.7" 
#define PACKAGE	"xccs"

using namespace std;

namespace SCIRun {

// ****************************************************************************
// ************************ Class: SocketConnection ***************************
// ****************************************************************************

class SocketConnection
{

public:
  SocketConnection();
  ~SocketConnection();

  int get_stream( string hostname, int port, string file_read );

private:

};

} // End namespace SCIRun
 
#endif // SocketConnection_h



