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
 * C++ (CC) FILE : SocketConnection.cc
 *
 * DESCRIPTION   : This represents the first, rough iteration of a wrapper
 *                 class for socket code in SCIRun.  The socket code from
 *                 the DDDAS StreamReader module has been added as a method
 *                 in this class.  No other infrastructure has been developed
 *                 for the class yet. 
 *                     
 * AUTHOR(S)     : Jenny Simpson
 *                 SCI Institute
 *                 University of Utah
 *
 *                 Chad Shannon
 *      	   Center for Computational Sciences
 *                 University of Kentucky
 * 
 * CREATED       : Mon Mar  1 16:33:34 MST 2004
 * MODIFIED      : Mon Mar  1 16:33:34 MST 2004
 * DOCUMENTATION :
 * NOTES         : Much of the code contained in this class has been 
 *                 borrowed from the xccs package created by Chad Shannon. 
 *                 Most of the xccs code was stripped from XMMS-1.2.7 source 
 *                 code.  In any case, it is just basic socket code.
 *
 * Copyright (C) 2003 SCI Group
*/
 
// SCIRun includes
#include <Packages/DDDAS/Core/Utils/SocketConnection.h>
 
namespace SCIRun {

/*===========================================================================*/
// 
// SocketConnection
//
// Description : Constructor
//
// Arguments   : none
//
SocketConnection::SocketConnection()
{
}

/*===========================================================================*/
// 
// ~SocketConnection
//
// Description : Destructor
//
// Arguments   : none
//
SocketConnection::~SocketConnection()
{
}

/*===========================================================================*/
// 
// get_stream
//
// Description : This is the xccs stream reading code used by the DDDAS 
//               StreamReader module.  It uses low-level socket code to open 
//               a connection to an mp3 file being streamed on a specified 
//               host and port.  Returns the socket descriptor for the stream 
//               which can then be used to read the stream.  Returns -1 on 
//               failure.
//
// Arguments   : 
//
// string hostname - Name of host to connect to.
//
// int port - Port to connect to on host.
//
// string file_read - MP3 file or other stream file to connect to. 
//
int SocketConnection::get_stream( string hostname, int port, string file_read )
{
  // This code is taken almost entirely from xccs
  char host[80], filename[80];
  char url[80];
  char file[80];
  char temp[128];
  char *chost;
  int error_num, err_len, cport;
  fd_set set; // File descriptor set
  struct hostent *hp; // Host entry
  struct sockaddr_in address; 
  struct timeval tv;
  int going = 1;
  int sock;

  // Set up host, port, url, and filename, and file variables
  strcpy( host, hostname.c_str() );
  stringstream s;
  s << port;
  string port_str = s.str();
  string url_str = "http://" + hostname + ":" + port_str + "/" + file_read;   
  strcpy( url, url_str.c_str() );
  strcpy( filename, file_read.c_str() );
  string file_str = "/" + file_read;
  strcpy( file, file_str.c_str() );
  chost = host;
  cport =  port;

  // Initialize socket descriptor
  // This is a TCP socket
  sock = socket( AF_INET, SOCK_STREAM, 0 );
  //fcntl(sock, F_SETFL, O_NONBLOCK);  // Sets socket to non-blocking
  address.sin_family = AF_INET;

  cout << "(SocketConnection::get_stream) LOOKING UP " << chost << std::endl;

  if( !(hp = gethostbyname(chost)) )
  {
    cerr << "(SocketConnection::get_stream) ERROR: Couldn't look up host "
         << chost << std::endl;
    return -1;
  }

  memcpy( &address.sin_addr.s_addr, *(hp->h_addr_list), 
          sizeof(address.sin_addr.s_addr) );
  address.sin_port = htons( cport );

  cout << "(SocketConnection::get_stream) CONNECTING TO " << chost << ":" 
       << cport << std::endl;

  if( connect(sock, (struct sockaddr *) &address, sizeof (struct sockaddr_in))
       == -1 )
  {
    if( errno != EINPROGRESS )
    {
      cerr << "(SocketConnection::get_stream) ERROR: Couldn't connect to host "
           << chost << " connect failed" << std::endl;
      return -1;
    }
  }

  while( going )
  {
    tv.tv_sec = 0;
    tv.tv_usec = 10000;
    FD_ZERO(&set);
    FD_SET(sock, &set);
    if( select(sock + 1, NULL, &set, NULL, &tv) > 0 )
    {
      err_len = sizeof( error_num );
      getsockopt( sock, SOL_SOCKET, SO_ERROR, &error_num, 
                  (socklen_t *) &err_len );
      if( error_num && errno != EINPROGRESS )
      {
	cerr << "(SocketConnection::get_stream) ERROR: Couldn't connect to host " 
             << chost << ", getsockopt failed" << std::endl;
        perror( NULL );
	return -1;
					
      }
      break;
    }
  }

  sprintf( temp,
           "GET %s HTTP/1.0\r\nHost: %s\r\nUser-Agent: %s/%s\r\n%s%s%s%s\r\n", 
           file, host, PACKAGE, VERSION, "", "", "", "" );
				
  write( sock, temp, strlen(temp) );
	
  cout << "(SocketConnection::get_stream) CONNECTED: WAITING FOR REPLY" 
       << std::endl;

  return sock;
}

} // End namespace SCIRun
