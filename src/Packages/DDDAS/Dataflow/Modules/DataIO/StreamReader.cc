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
 * C++ (CC) FILE : StreamReader.cc
 *
 * DESCRIPTION   : Connects to a remote host, receives streaming data and 
 *                 buffers the data between specified headers.  Currently 
 *                 this data is LatVol mesh data, so once the data for a
 *                 single mesh has been buffered, an actual LatVol mesh is 
 *                 constructed and sent down the SCIRun pipeline to the 
 *                 Viewer.  There is some data loss between reads because of 
 *                 both network latency and processing that necessarily 
 *                 occurs between reads.  This data loss is ignored for now
 *                 since the solution headers aren't frequent and 
 *                 informational enough to determine which chunks of data 
 *                 have been lost so that they can be replaced.  As a result,
 *                 the LatVol meshes produced almost always have some degree 
 *                 of error and this shows up as nodes in the mesh being
 *                 shifted from their correct position. 
 *                     
 * AUTHOR(S)     : Chad Shannon
 *      	   Center for Computational Sciences, University of Kentucky
 *	           Copyright 2003
 *
 *                 Jenny Simpson
 *                 SCI Institute
 *                 University of Utah
 *         
 * CREATED       : 7/9/2003
 *
 * MODIFIED      : Mon Aug  4 09:20:52 MDT 2003
 *
 * DOCUMENTATION :
 * 
 * NOTES         : Most of the code contained in this module has been 
 *                 borrowed from the SampleLattice module created by Mike 
 *                 Callahan and the xccs package created by Chad Shannon. Most 
 *                 of the xccs code was stripped from XMMS-1.2.7 source code.
 *
 * Copyright (C) 2003 SCI Group
*/
 
// SCIRun includes

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <Packages/DDDAS/share/share.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/ConditionVariable.h>

// Standard lib includes

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

#define BUFFER_SIZE	4096
#define VERSION	"0.0.7" 
#define PACKAGE	"xccs"

namespace DDDAS {

using namespace SCIRun;


// ****************************************************************************
// ***************************** Class: StreamReader **************************
// ****************************************************************************
   
//! Continually reads and processes data from an mp3 stream
class DDDASSHARE StreamReader : public Module {

public:

    
  //! Virtual interface

  StreamReader(GuiContext* ctx);

  virtual ~StreamReader();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  //! General functions for stream reading and processing

  int find_string( istringstream& input, string str ) const;

  int get_stream();

  void kill_helper( Thread * helper_thread );

  void process_stream();

  void read_stream();

  void remove_headers( unsigned char * old_buffer, unsigned char * new_buffer,
                       int& num_bytes );

  int seek_header( istringstream& input, int sock, unsigned char * buffer, 
                   string * headers, int num_headers, string & header_name,
                   int & nread );

  int update_input( istringstream& input, int sock, unsigned char * buffer );

  //! Functions specific to LatVol mesh data

  void fill_mesh( double * sol_pts, int num_sols ) const;

  int get_sols( istringstream& input, double * sol_pts, int num_sols );

  void process_mesh( istringstream& input );

  void read_mesh( istringstream& input, int sock, unsigned char * buffer, 
                  int start, ssize_t nread );

    
private:

  //! GUI variables
  GuiString hostname_;
  GuiInt port_;
  GuiString file_read_;
  GuiString file_write_;
  GuiInt stop_sr_;

  //! Thread safety
  ConditionVariable     stream_cond_;
  Mutex                 buffer_lock_;

  FieldOPort *ofp_;
  unsigned char * final_buffer_;

};


// ****************************************************************************
// ************************** Class: ReaderThread *****************************
// ****************************************************************************

//! Thread that continually reads and buffers data from an mp3 stream, looks
//! for headers in the data and sends the data to the processing thread when 
//! these headers are found.
class DDDASSHARE ReaderThread : public Runnable {
public:
  ReaderThread(StreamReader * sr );
  virtual ~ReaderThread();
  virtual void run();

private:
  StreamReader * stream_reader_;
};

/*===========================================================================*/
// 
// Description : Constructor
//
// Arguments   :
//
// StreamReader * sr - A pointer to the StreamReader object that this reader 
//                     thread is to be associated with.
//
ReaderThread::ReaderThread( StreamReader * sr ) :
  stream_reader_(sr) 
{
}

/*===========================================================================*/
// 
// Description : Destructor
//
// Arguments   : none
//
ReaderThread::~ReaderThread()
{
}

/*===========================================================================*/
// 
// Description : This is essentially a callback that gets called when the 
//               reader thread is initialized.  In this case the thread begins
//               reading a stream when it is initialized.
//
// Arguments   : none
//
void
ReaderThread::run()
{
  cout << "(ReaderThread::run) I'm running!\n";

  // Read and parse the solution points contained in the data from the stream
  stream_reader_->read_stream();
} 


// ****************************************************************************
// ************************** Class: ProcessorThread **************************
// ****************************************************************************

//! Processes data that has been read from an mp3 stream by the reader thread.
class DDDASSHARE ProcessorThread : public Runnable {
public:
  ProcessorThread(StreamReader * sr );
  virtual ~ProcessorThread();
  virtual void run();

private:
  StreamReader * stream_reader_;
};

/*===========================================================================*/
// 
// Description : Constructor
//
// Arguments   :
//
// StreamReader * sr - A pointer to the StreamReader object that this 
//                     processor thread is to be associated with.
//
ProcessorThread::ProcessorThread( StreamReader * sr ) :
  stream_reader_(sr) 
{
}

/*===========================================================================*/
// 
// Description : Destructor
//
// Arguments   : none
//
ProcessorThread::~ProcessorThread()
{
}

/*===========================================================================*/
// 
// Description : This is essentially a callback that gets called when the 
//               reader thread is initialized.  In this case the thread begins
//               waiting for available stream data to process as soon as it
//               is initialized.
//
// Arguments   : none
//
void
ProcessorThread::run()
{
  cout << "(ProcessorThread::run) I'm running!\n";

  // Read and parse the solution points contained in the data from the stream
  stream_reader_->process_stream();
} 

 
DECLARE_MAKER(StreamReader)

/*===========================================================================*/
// 
// Description : Constructor
//
// Arguments   : 
//
// GuiContext* ctx - GUI context
//
StreamReader::StreamReader(GuiContext* ctx)
  : Module("StreamReader", ctx, Source, "DataIO", "DDDAS"),
    hostname_(ctx->subVar("hostname")),   
    port_(ctx->subVar("port")),   
    file_read_(ctx->subVar("file-read")),   
    file_write_(ctx->subVar("file-write")),
    stop_sr_(ctx->subVar("stop-sr")),
    stream_cond_("StreamReader: waits for stream reading/processing to finish."),
    buffer_lock_("StreamReader: controls mutable access to the buffer.")  
{  
}


/*===========================================================================*/
// 
// Description : Destructor
//
// Arguments   : none
//
StreamReader::~StreamReader()
{
}

/*===========================================================================*/
// 
// Description : The execute function for this module.  This is the control
//               center for the module.
//
// Arguments   : none
//
void
StreamReader::execute()
{
  // Declare output field
  ofp_ = (FieldOPort *)get_oport("Output Sample Field");
  if (!ofp_) {
    error("(StreamReader::execute) Unable to initialize oport 'Output Sample Field'.");
    cerr << "(StreamReader::execute) Unable to initialize oport 'Output Sample Field'.\n";
    return;
  }

  // Create two threads, one that reads and caches away the data.
  // Another that checks for a complete mesh status, and sends it downstream.

  Runnable * r = new ReaderThread( this );
  Thread * reader_thread =  new Thread( r, "reader" );

  Runnable * p = new ProcessorThread( this );
  Thread * proc_thread =  new Thread( p, "processor" );

}

/*===========================================================================*/
// 
// Description : The tcl_command function for this module.
//
// Arguments   :
//
// GuiArgs& args - GUI arguments
//
// void* userdata - ???
// 
void
StreamReader::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

/*===========================================================================*/
// 
// Description : Reads an input stream until a specified string is encountered 
//               or the input ends.  If the string is encountered, returns the 
//               index of the string in the internal buffer.  Otherwise, 
//               returns -1.
//
// Arguments   :
//
// istringstream& input - The input stream that is to be searched for the given
//                        string
//
// string str - The string to find within the given input stream 
//
int 
StreamReader::find_string( istringstream& input, string str ) const
{
  cout << "(StreamReader::find_string) Inside\n";
  string s;
  while( input >> s )
  {
    if( s == str )
    {
      int pos = input.tellg();
      return pos - 8;
    }
  }

  cout << "(StreamReader::find_string) Leaving\n";

  // The string was not found, return -1
  return -1;
}

/*===========================================================================*/
// 
// Description : This is the xccs stream reading code provided by Chad.  It 
//               uses low-level socket code to open an mp3 file on a specified 
//               port and host.  Returns the socket descriptor for the stream 
//               which can then be used to read the stream.  Returns -1 on 
//               failure.
//
// Arguments   : none
//
int
StreamReader::get_stream()
{
  // This code is taken almost entirely from xccs
  char host[80], filename[80];
  char url[80];
  char file[80];
  char temp[128];
  char *chost;
  int error, err_len, cport;
  fd_set set; // File descriptor set
  struct hostent *hp; // Host entry
  struct sockaddr_in address; 
  struct timeval tv;
  int going = 1;
  int sock;

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
  chost = host;
  cport =  port;

  // Initialize socket descriptor
  sock = socket(AF_INET, SOCK_STREAM, 0);
  //fcntl(sock, F_SETFL, O_NONBLOCK);
  address.sin_family = AF_INET;

  cout << "(StreamReader::get_stream) LOOKING UP " << chost << "\n";

  if (!(hp = gethostbyname(chost)))
  {
    cerr << "Couldn't look up host " << chost << "\n";
    return -1;
  }

  memcpy(&address.sin_addr.s_addr, *(hp->h_addr_list), sizeof (address.sin_addr.s_addr));
  address.sin_port = htons(cport);

  cout << "(StreamReader::get_stream) CONNECTING TO " << chost << ":" << cport << "\n";

  if (connect(sock, (struct sockaddr *) &address, sizeof (struct sockaddr_in)) == -1)
  {
    if (errno != EINPROGRESS)
    {
      cerr << "(StreamReader::get_stream) Couldn't connect to host " << chost 
           << " connect failed\n";
      return -1;
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
	cerr << "(StreamReader::get_stream) Couldn't connect to host " 
             << chost << ", getsockopt failed\n";
        perror(NULL);
	exit(1);
					
      }
      break;
    }
  }

  sprintf(temp,"GET %s HTTP/1.0\r\nHost: %s\r\nUser-Agent: %s/%s\r\n%s%s%s%s\r\n", file, host, PACKAGE, VERSION, "", "", "", "");
				
  write(sock, temp, strlen(temp));
	
  cout << "(StreamReader::get_stream) CONNECTED: WAITING FOR REPLY\n";

  return sock;
}

/*===========================================================================*/
// 
// Description : Kill a specified helper thread.
//
// Arguments   :
//
// Thread * helper_thread - A pointed to the thread to be killed
//
void
StreamReader::kill_helper( Thread * helper_thread )  
{
  // kill the helper thread
  if (helper_thread)
  {
    helper_thread->join();
    helper_thread = 0;
  }
}

/*===========================================================================*/
// 
// Description : Parses the contents of a given buffer of data that 
//               has been read from the stream.  Figures out what kind of data
//               is contained in the buffer and passes the data to the 
//               appropriate function for processing.
//
// Arguments   : none
//
void
StreamReader::process_stream()
{
  cout << "(StreamReader::process_stream) Inside\n";

  // Continually wait for the reader to set the condition signal, or for the 
  // user to click the "Stop" button on the UI
  int stop_sr =stop_sr_.get() ;
  while( !stop_sr )
  {

    cout << "(StreamReader::process_stream) Waiting for condition variable\n";

    // Wait for condition signal from reader.  The condition signal indicates 
    // that the reader has buffered enough data for processing.
    stream_cond_.wait(buffer_lock_);

    // We got the hand-off from the reader, so now we begin processing the 
    // buffer data

    // Convert buffer to input stream since a stream is easier to parse
    string str = (const char *) final_buffer_;
    istringstream input( str, istringstream::in );

    // Deallocate final buffer memory
    if( final_buffer_ != 0 )
    {
      delete [] final_buffer_;
    }

    // Grab the first string in the input stream
    input >> str;

    // Check to see what kind of data to process.  This is indicated by the
    // first string in the buffer.
    if( str == "solution" )
    {
      // This is mesh data containing solution points
      process_mesh( input );
    }
    else
    {
      // Got unrecognized data, spit out an error message and return 
      cerr << "(StreamReader::process_stream) WARNING: Unrecognized header\n";
      cout << "(StreamReader::process_stream) str = " << str << "\n";
      return;
    }

    // Update stop variable
    stop_sr_ = ctx->subVar("stop-sr");      
    stop_sr = stop_sr_.get();  

  }

  cout << "(StreamReader::process_stream) done\n";
  cout << "(StreamReader::process_stream) Leaving\n";
}

/*===========================================================================*/
// 
// Description : Continually read from the mp3 stream, remove mp3 headers, 
//               and check for recognized headers.  When the first recognized
//               header is found, call the appropriate function to process
//               that data type (i.e. mesh data).  When the processing funtion
//               returns (i.e. a second solution header is encountered), update
//               the condition variable to wake up the other thread so that it 
//               will check the data. Note that the buffer variable must be 
//               locked and unlocked so that reads and writes don't conflict.
// 
// Arguments   : none
//
void 
StreamReader::read_stream() 
{
  // Get stream socket
  int sock = get_stream();

  int cnt = 0;
  cout << "(StreamReader::read_stream) Receiving data.....\n";

  // Initialize buffer to store input
  unsigned char buffer[BUFFER_SIZE];

  int stop_sr =stop_sr_.get() ;
  istringstream input( istringstream::in );

  // Continually read full datasets until told to stop
  while( !stop_sr )
  {

    // If this is the first time reading, need to lock mutex
    if( cnt == 0 )
    {
      cout << "(StreamReader::read_stream) Trying to lock mutex\n"; 
      // Lock buffer mutex
      buffer_lock_.lock();
    }

    // Continually read small chunks off of the stream until the first header
    // is found
    int num_headers = 1;
    string headers[1] = {"solution"};
    string header_name = "NONE";
    int nread = 0;

    // Read until an appropriate header is found
    int start = seek_header( input, sock, buffer, headers, num_headers, header_name, nread );

    // Call appropriate read function for this type of data
    if( header_name == "solution" ) 
    {
      read_mesh( input, sock, buffer, start, nread );
    }

    // Unlock buffer mutex
    cout << "(StreamReader::read_stream) Unlocking mutex\n";
    buffer_lock_.unlock();

    // Signal to processing thread that buffer is available
    cout << "(StreamReader::read_stream) Broadcasting condition signal\n";
    stream_cond_.conditionBroadcast();

    // Wait for condition signal from processor
    cout << "(StreamReader::read_stream) Waiting for condition signal\n";
    stream_cond_.wait(buffer_lock_);

    cnt++;
  
    // Update the stop variable
    stop_sr_ = ctx->subVar("stop-sr");      
    stop_sr = stop_sr_.get();
  }	

  cout << "(StreamReader::read_stream) Leaving\n";  
  close( sock );

}

/*===========================================================================*/
// 
// Description : Removes all mp3 headers (which occur every 417 bytes) from a 
//               buffer.  Returns the modified buffer as new_buffer and the 
//               buffer size by reference as num_bytes. Fills the remaining 
//               bytes in old_buffer and new_buffer with zeros.  Leaves the 
//               mp3 headers in old_buffer.  Assumes that both old_buffer and 
//               new_buffer are of size BUFFER_SIZE.
//
// Arguments   :
//
// unsigned char * old_buffer - The original buffer that contains mp3 headers
//                              that need to be removed.
//  
// unsigned char * new_buffer - The new buffer that gets assigned the old 
//                              buffer data minus the mp3 headers
//
// int& num_bytes - Number of bytes to examine in old_buffer.  Gets modified
//                  to contains the number of bytes in new_buffer after the mp3
//                  headers have been stripped.
//
void
StreamReader::remove_headers( unsigned char * old_buffer, 
                              unsigned char * new_buffer, int& num_bytes )  
{
  // Grab some extra bytes just in case
  int num = num_bytes;
  num_bytes = 0; 
  for( int i = 0; i < num; i++ )
  {
    unsigned char ch = old_buffer[i];

    if( ch == 0xFF  ) 
    {
      // This is the first byte of the mp3 header, skip the header (4 bytes)
      i += 3;
    }
    else
    {
      new_buffer[num_bytes] = ch;
      num_bytes++;
    }
  }

  // Zero out the rest of the buffers
  for( int j = num_bytes; j < BUFFER_SIZE; j++ )
  {
    new_buffer[j] = 0;
  }

  // Zero out the rest of the buffers
  for( int k = num; k < BUFFER_SIZE; k++ )
  {
    old_buffer[k] = 0;
  }

}

/*===========================================================================*/
// 
// Description : Reads data from a stream until a string is found that matches 
//               one of the entries in the array of header strings.  Returns
//               the internal buffer index where the header string begins.
//
// Arguments   :
//
// istringstream& input - Input string stream to search for a valid header
//
// int sock - Socket descriptor for an mp3 stream
//
// unsigned char * buffer - Buffer to store char data from socket reads
//
// string * headers - Array of valid header strings to look for
//
// int num_headers - The number of header strings contained in the 'headers'
//                   array
//
// string & header_name, 
//
// int & nread - Returned by reference as the number of bytes read in the last
//               read
//
int
StreamReader::seek_header( istringstream& input, int sock, 
                           unsigned char * buffer,  
                           string * headers, int num_headers, 
                           string & header_name, int & nread ) 
{
  int stop_sr =stop_sr_.get() ;

  // Continually read small chunks off of the stream until the first header
  // is found
  while( !stop_sr )
  {
    // Read chunk of data from stream
    nread = update_input( input, sock, buffer );

    // Check for an appropriate header in this chunk of data
    string s;
    while( input >> s )
    {
      // Check each string to see if it matches any of the valid headers
      for( int i = 0; i < num_headers; i++ ) 
      {
        if( s == headers[i] )
        {
          header_name = headers[i];
          int pos = input.tellg();
          assert( (pos - 8) < nread );
          return pos - 8;
        }
      }
    } 

    // Update the stop variable
    stop_sr_ = ctx->subVar("stop-sr");      
    stop_sr = stop_sr_.get();
  }
  
  // Should never get here
  return -1;
}

/*===========================================================================*/
// 
// Description : Reads the next chunk of data off of the mp3 stream, removes 
//               the mp3 headers, and updates the input stream and buffer so 
//               that they contain the updated data.  Returns the number of 
//               bytes remaining in buffer after mp3 headers have been removed.
//               Assumes that buffer is of size BUFFER_SIZE.
//
// Arguments   :
//
// istringstream& input - Empty input stream that gets assigned the data read
//                        in from the mp3 stream
// 
// int sock - Socket descriptor for the mp3 stream
//
// unsigned char * buffer - Character buffer of size BUFFER_SIZE that is used
//                          to buffer data read from the mp3 stream.  Data
//                          in this buffer gets copied to the input string 
//                          stream.
//
int 
StreamReader::update_input( istringstream& input, int sock, unsigned char * buffer ) 
{
  
  cout << "(StreamReader::update_input) Inside\n"; 

  int nread = 0;
  // Read chunck of data from stream
  if( (nread = read(sock, buffer, BUFFER_SIZE)) < 0 )
  {
    cerr << "(StreamReader::update_input) Read failed\n";  
    perror(NULL);
    return nread; 
  }
  else if( nread == 0 )
  {
    // End of file
    cerr << "(StreamReader::update_input) Read 0 bytes\n";
    return nread;
  }

  cout << "(StreamReader::update_input) Read succeeded, nread = " << nread 
       << "\n";

  cout << "(StreamReader::update_input) Removing headers\n"; 

  // Remove all mp3 headers, create new buffer without the headers
  unsigned char new_buffer[BUFFER_SIZE];
  remove_headers( buffer, new_buffer, nread );

  // Update buffer
  memcpy( buffer, new_buffer, BUFFER_SIZE );

  cout << "(StreamReader::update_input) Modifying new input\n"; 

  // Convert character buffer to istringstream
  string str_buf = (const char *) buffer;
  input.clear();
  input.str( str_buf );

  cout << "(StreamReader::update_input) Done modifying new input\n"; 

  if ( !input )
  {
    error( "(StreamReader::update_input)  Failed to open stream." );
    cerr << "(StreamReader::update_input)  Failed to open stream.\n";
  }    
  
  assert( nread <= BUFFER_SIZE );

  cout << "(StreamReader::update_input) Leaving\n"; 
  return nread;
}

/*===========================================================================*/
// 
// Description : Reads solution points from an array, populates a LatVol mesh 
//               with them, and sends the mesh down the SCIRun pipeline to be 
//               eventually visualized by the Viewer.
//
// Arguments   :
// 
// double * sol_pts - Array of doubles, where each entry in the array 
//                    represents a solution point which is the value of a 
//                    specific node in a LatVol mesh
//
// int num_sols - Number of entries / solution points in the sol_pts array
//
void
StreamReader::fill_mesh( double * sol_pts, int num_sols ) const
{
  Point minb, maxb;
  minb = Point(-1.0, -1.0, -1.0);
  maxb = Point(1.0, 1.0, 1.0);

  Vector diag((maxb.asVector() - minb.asVector()) * (0.0/100.0));
  minb -= diag;
  maxb += diag;
  
  if( sol_pts == 0 ) return;

  // Create blank mesh.
  int cube_root = (int) cbrt( num_sols );
  unsigned int sizex;
  unsigned int sizey;
  unsigned int sizez;
  sizex = sizey = sizez = Max(2, cube_root) + 1;
  LatVolMeshHandle mesh = scinew LatVolMesh(sizex, sizey, sizez, minb, maxb);

  // Assign data to cell centers
  Field::data_location data_at = Field::CELL;

  // Create Image Field.
  FieldHandle ofh;

  LatVolField<double> *lvf = scinew LatVolField<double>(mesh, data_at);
  if (data_at != Field::NONE)
  {
    LatVolField<double>::fdata_type::iterator itr = lvf->fdata().begin();

    // Iterator for solution points array
    int i = 0; 
    while (itr != lvf->fdata().end())
    {
      assert( i < num_sols );
      *itr = sol_pts[i];
      ++itr;
      i++;
    }
  } 
  ofh = lvf;

  // Send data to output field  
  ofp_->send_intermediate(ofh);

}

/*===========================================================================*/
// 
// Description : Reads in the solution points from a stream until all
//               of the solution points have been read or an error has 
//               occurred.  Stores the solution points in the sol_pts array.
//               Returns the number of solution points read.
//
// Arguments   :
//
// istringstream& input - input stream containing a series of doubles that
//                        represent solution points
//
// double * sol_pts - a previously allocated array that gets populated with 
//                    solution points
//
// int num_sols - size of the sol_pts array and the number of solution points
//                to read in
//
int 
StreamReader::get_sols( istringstream& input, double * sol_pts, int num_sols ) 
{
  int num_read = 0;
  double sol;
  char ch;

  int stop_sr = stop_sr_.get() ;

  // Read until we get all solution points or encounter an error
  while( num_read < num_sols && !stop_sr)
  {
    cout << "(StreamReader::get_sols) Reading solution points\n";
    if( input >> sol ) // I can read a double
    {

          // Store the double in the solution array
          sol_pts[num_read] = sol;
          cout << "(StreamReader::get_sols) sol_pts[" << num_read 
               << "] = " << sol_pts[num_read] << "\n";
          num_read++;  
    }
    else if( (ch = input.peek()) == 's' ) // I've hit mesh header
    {
      // If the string is another mesh header, check to see if we read
      // all the solution points we needed
      cout << "(StreamReader::get_sols) Got another solution header prematurely\n";
      break;
    } 
    else if( input.eof() ) // I've reached the end of the input stream 
    {
      cout << "(StreamReader::get_sols) Reached end of input stream\n";
      break;
    } 
    else  // I've reached something I don't recognize
    { 
      cerr << "(StreamReader::get_sols) ERROR: Input '" << ch 
           << "' unrecognized\n"; 

       // Just discard this for now
      if( input.fail() )
      {
        cerr << "(StreamReader::get_sols) WARNING: Input stream failed\n";
        input.clear();
      }   
      char char_ptr[100];
      input.getline( char_ptr, 100 );
      cout << "(StreamReader::get_sols) char_ptr = " << char_ptr << "\n";
      //break;
    }

    // Update stop variable
    stop_sr_ = ctx->subVar("stop-sr");      
    stop_sr = stop_sr_.get();  
 
  }

  return num_sols;
}

/*===========================================================================*/
// 
// Description : Parse buffered data that contains solution points for a 
//               LatVol mesh.  Populates an array with all of the solution 
//               points for a single mesh and then constructs a LatVol mesh 
//               that is eventually passed downstream to the Viewer.
//     
// Arguments   :
//
// istringstream& input - An input stream that begins with a solution header
//                        that is followed by a list of solution points that
//                        represent the data for one LatVol mesh
//
void 
StreamReader::process_mesh( istringstream& input ) 
{
    /*
    Here's an sample of what the buffer data should look like:

    Data Format
    ----------------

    [char *s (file name)]
    solution u -size=[int Ncells (number of solution pts)] -components=1 -type=nodal
    [double solution(0) (Solution point 0.  See notes below)]
    [double solution(1)]
    [double solution(2)]
    [double solution(3)]
    [double solution(4)]
    [double solution(5)]
    ...
    [double solution(Ncells)]

    Sample Data
    -----------

    sat.out
    solution u  -size=64000 -components=1 -type=nodal
    0.584279
    0.249236
    0.0711161
    0.0134137
    0.00190625
    0.000223068
    2.70226e-05
    ...
    
  */

  // Set up variables corresponding to the file values
  string filename_h;
  string solution;
  int components;
  string type;
  string str;
  char ch;
  int num_sols = 0;
   
  cout << "(StreamReader::process_mesh) Got beginning of solution set\n";  

  // Parse the header, assigning variable values
  input >> solution 
    >> ch >> ch >> ch >> ch >> ch >> ch 
    >> num_sols
    >> ch >> ch >> ch >> ch >> ch >> ch 
    >> ch >> ch >> ch >> ch >> ch >> ch
    >> components
    >> ch >> ch >> ch >> ch >> ch >> ch
    >> type;

  cout << "(StreamReader::process_mesh) num_sols = " << num_sols << "\n";  

  // Allocate memory for solution points
  double * sol_pts = new double[num_sols]; // Replace new with scinew?

  // Initialize the field values array with solution points with value 0
  memset( sol_pts, 0, num_sols );
        
  // Read the solution points from the input stream
  int num_read = get_sols( input, sol_pts, num_sols );

  // Do some error checking to make sure we got a valid mesh
  if( num_read == num_sols )
  {
    if( (ch = input.peek()) != 's' )
    {
      cerr << "(StreamReader::process_mesh) WARNING: Next value isn't header\n"; 
    }

    cout << "(StreamReader::process_mesh) Filling mesh\n"; 

    cout << "(StreamReader::process_mesh) Unlocking mutex\n"; 

    // Unlock buffer mutex
    buffer_lock_.unlock();

    cout << "(StreamReader::process_mesh) Broadcasting condition signal\n"; 
    // Signal to reader thread that buffer is available
    stream_cond_.conditionBroadcast();

    // Fill the mesh with these values
    fill_mesh( sol_pts, num_sols );

  }
  else
  {
    cout << "(StreamReader::process_mesh) Got incomplete mesh, discarding\n"; 
  }

  // Deallocate array
  delete [] sol_pts;

}

/*===========================================================================*/
// 
// Description : Reads all of the data for one mesh off of the mp3 stream 
//               starting from the beginning of a mesh header.
//
// Arguments   :
//
// istringstream& input - Input stream with the first chuck of data for a mesh
//
// int sock - Socket descriptor for the mp3 stream
//
// unsigned char * buffer - Buffer of char data that contains the last chunk
//                          of data read from the mp3 stream
//
// int start - The index in buffer where the mesh header occurs
// 
// ssize_t nread - Number of bytes in the last chunk of data read from the mp3 
//                 stream 
//
void
StreamReader::read_mesh( istringstream& input, int sock, 
                         unsigned char * buffer, int start, ssize_t nread ) 
{
  
  // Parse the header to get number of solutions
  char ch;
  int num_sols;
  input >> ch >> ch >> ch >> ch >> ch >> ch >> ch >> num_sols;

  cout << "(StreamReader::read_mesh) num_sols = " << num_sols << "\n";

  // Make new buffer to fit all of the solution points and header
  int final_size = 100 + num_sols * 13;
  final_buffer_ = new unsigned char[final_size];

  // Set all values of final buffer to 0
  memset( final_buffer_, 0, final_size );

  cout << "(StreamReader::read_mesh) final_size = " << final_size << "\n";
  cout << "(StreamReader::read_mesh) start = " << start << "\n";
      
  // Copy the first section of the mesh into the final buffer
  int cpy_size = nread - start;
  int offset = cpy_size;
  assert( cpy_size > 0 && cpy_size <= final_size 
          && cpy_size <= (BUFFER_SIZE - start) );

  memcpy( final_buffer_, buffer + start, cpy_size );      

  int stop_sr = stop_sr_.get() ;

  // Read until the next solution header is found, appending new data to 
  // the end of the buffer.
  while( !stop_sr )
  {
    assert( offset < final_size );

    // Do another read
    nread = update_input( input, sock, buffer );

    // Check for another solution header
    if( (start = find_string(input, "solution")) >= 0 )
    {
      cout << "(StreamReader::read_mesh) Found second header\n";
      // Copy data up until next header into final buffer
      assert( start < nread && start <= (final_size - offset) );
      memcpy( final_buffer_ + offset, buffer, start ); 
      break;
    }
  
    // Append new data to the end of the final buffer
    if( nread > (final_size - offset) )
    {
      cerr << "(StreamReader::read_mesh) WARNING: Filled buffer before" 
           << " finding another solution header\n";
      // Fill remainder of final_buffer_
      memcpy( final_buffer_ + offset, buffer, (final_size - offset) );      
      break;
    }

    memcpy( final_buffer_ + offset, buffer, nread );      
    offset += nread;     
        
    // Update stop variable
    stop_sr_ = ctx->subVar("stop-sr");      
    stop_sr = stop_sr_.get();
  } 

}

} // End namespace DDDAS




