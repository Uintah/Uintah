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
 * StreamReader.cc:  Connects to a remote host, receives and saves data to 
 *                   hard-disk.
 *
 * Author: 
 *         Chad Shannon
 * 	   Center for Computational Sciences, University of Kentucky
 *	   Copyright 2003
 *
 * Modified by: 
 *         Jenny Simpson
 *         SCI Institute
 *         University of Utah
 *         6/24/2003
 * 
 * Description:  This program connects, receives and saves
 *	   data to hard-disk.  Here is a basic outline of the function of 
 *         this code:
 *         - Connects to a port on some host (error if no connect)
 *         - Sends http get request to remote host, asking for an mp3 
 *           file
 *         - Reads data from socket until MAX_COUNT is reached, puts this data 
 *           in a buffer
 *         - Once MAX_COUNT is reached, writes buffer to a file
 *           with .txt extension.
 *
 * Notes:  Eventually 
 *	   the data will be split into seperate files by placing a marker
 *	   within the header ( like using the CRC bit to flag EOF).
 *
 * 	   Just so I (Chad) don't get sued, most of this code was stripped 
 *         from XMMS-1.2.7 source code.
 *     
 *         Additional comments added by Jenny Simpson <simpson@cs.utah.edu>
 *         on 6/23/2003
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <Packages/DDDAS/share/share.h>

#include <iostream>
#include <fstream>
#include <assert.h>
#include <sys/types.h>
#include <dirent.h>

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

#define MAX_COUNT 	10000
#define BUFFER_SIZE	1024
#define VERSION	"0.0.7" 
#define PACKAGE	"xccs"

namespace DDDAS {

using namespace SCIRun;
  
class DDDASSHARE StreamReader : public Module {

public:
  StreamReader(GuiContext* ctx);
  virtual ~StreamReader();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  GuiString hostname_;
  GuiInt port_;
  GuiString file_read_;
  GuiString file_write_;

};
 
 
DECLARE_MAKER(StreamReader)

StreamReader::StreamReader(GuiContext* ctx)
  : Module("StreamReader", ctx, Source, "DataIO", "DDDAS"),
    hostname_(ctx->subVar("hostname")),   
    port_(ctx->subVar("port")),   
    file_read_(ctx->subVar("file-read")),   
    file_write_(ctx->subVar("file-write"))   
{  
}



StreamReader::~StreamReader()
{
}


void
StreamReader::execute()
{
  // This code is taken almost entirely from xccs
  char host[80], filename[80];
  char url[80];
  char file[80];
  char temp[128];
  char *chost;
  int cnt, written, error, err_len, cport;
  fd_set set; // File descriptor set
  struct hostent *hp; // Host entry
  struct sockaddr_in address; 
  struct timeval tv;
  int going;
  int sock, rd_index, wr_index, buffer_length, prebuffer_length;
  unsigned char buffer[BUFFER_SIZE];
  FILE *output_file = NULL;

  // Get GUI variables
  string hostname = hostname_.get();
  int port = port_.get();
  string file_read = file_read_.get();
  string file_write = file_write_.get();

  // Set up host, port, url, and filename, and file variables
  strcpy(host, hostname.c_str());
  stringstream s;
  s << port;
  string port_str = s.str();
  string url_str = "http://" + hostname + ":" + port_str + "/" + file_read;   
  strcpy(url, url_str.c_str());
  strcpy(filename, file_read.c_str());
  string file_str = "/" + file_read;
  strcpy(file, file_str.c_str());


  /*if ((!filename || !*filename) && url[strlen(url) - 1] != '/')
		temp = g_strconcat(url, "/", NULL);
	else
		temp = g_strdup(url);
  */

  chost = host;
  cport =  port;

  sock = socket(AF_INET, SOCK_STREAM, 0);
  //fcntl(sock, F_SETFL, O_NONBLOCK);
  address.sin_family = AF_INET;

  printf("LOOKING UP %s\n", chost);

  if (!(hp = gethostbyname(chost)))
  {
    printf("Couldn't look up host %s\n", chost);
    exit(1);
  }

  memcpy(&address.sin_addr.s_addr, *(hp->h_addr_list), sizeof (address.sin_addr.s_addr));
  address.sin_port = htons(cport);

  printf("CONNECTING TO %s:%d\n", chost, cport);

  if (connect(sock, (struct sockaddr *) &address, sizeof (struct sockaddr_in)) == -1)
  {
    if (errno != EINPROGRESS)
    {
      printf("Couldn't connect to host %s, connect failed\n", chost);
      exit(1);
    }
  }

  while (going)
  {
    tv.tv_sec = 0;
    tv.tv_usec = 10000;
    FD_ZERO(&set);
    FD_SET(sock, &set);
    if (select(sock + 1, NULL, &set, NULL, &tv) > 0)
    {
      err_len = sizeof (error);
      getsockopt(sock, SOL_SOCKET, SO_ERROR, &error, (socklen_t *) &err_len);
      if (error && errno != EINPROGRESS)
      {
	printf("Couldn't connect to host %s, getsockopt failed\n", chost);
        perror(NULL);
	exit(1);
					
      }
      break;
    }
  }

  //file = g_strconcat("/", filename, NULL);
  sprintf(temp,"GET %s HTTP/1.0\r\nHost: %s\r\nUser-Agent: %s/%s\r\n%s%s%s%s\r\n", file, host, PACKAGE, VERSION, "", "", "", "");
				
  write(sock, temp, strlen(temp));
	
  printf("CONNECTED: WAITING FOR REPLY\n");

  output_file = fopen(file_write.c_str(), "wb");
		
  cnt = 0;
  going = 1;
  printf("Receiving data.....\n");

  ssize_t nread;
  while(going)
  {
    if( (nread = read(sock, buffer, BUFFER_SIZE)) < 0 )
    {
      printf("Read failed\n");  
      perror(NULL);
      break; 
    }
    else if( nread == 0 )
    {
      // End of file
      break;      
    }

    fwrite(buffer, sizeof(unsigned char), BUFFER_SIZE, output_file);
    cnt++;

    if( cnt > MAX_COUNT )
      going = 0;

    printf("Read number %d succeeded\n", cnt);
  }
	
  printf("done\n");
  close(sock);
  fclose(output_file);

}

void
StreamReader::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace DDDAS




