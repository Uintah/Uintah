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
 *                 processes the data.  Currently this streamed data comes in 
 *                 several formats and is intended to simulate sensor data.
 *
 *                 For the stream 2 spec, a point cloud mesh is built from the
 *                 scalar values and coordinates sent across the stream.
 *  
 *                 The stream 3 spec has not yet been tested, but there is some
 *                 code in place to be used as a starting point.
 * 
 *                 The places where code needs to be modified to accommodate
 *                 new stream data formats is indicated with '==>'.
 *                     
 * AUTHOR(S)     : Jenny Simpson
 *                 SCI Institute
 *                 University of Utah
 * 
 * CREATED       : 7/9/2003
 *
 * MODIFIED      : Thu Feb 26 15:15:13 MST 2004
 *
 * DOCUMENTATION :
 * 
 * Copyright (C) 2004 SCI Group
*/
 
// SCIRun includes
  
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <Packages/DDDAS/share/share.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h> 
#include <Core/Thread/ConditionVariable.h>
#include <Packages/DDDAS/Core/Datatypes/PointCloudWrapper.h>
#include <Packages/DDDAS/Core/Utils/SocketConnection.h>

// Standard lib includes

#include <iostream>
#include <fstream>
#include <assert.h>
#include <sys/types.h>

// Defines

#define MAX_NUM_VALS    64000 // Maximum number of values that can be sent for
                              // each sensor packet
//#define THREAD_ENABLED 1
#define MAX_ERROR_STRING 1024

namespace DDDAS {

using namespace SCIRun;


// ****************************************************************************
// ***************************** Class: StreamReader **************************
// ****************************************************************************

struct PointCloudValue {
  string id;
  Point pt;
  float data;
  string data_name;
};
   
//! Continually reads and processes data from an mp3 stream
class DDDASSHARE StreamReader : public Module {

public:
  
  //! Virtual interface

  StreamReader(GuiContext* ctx);

  virtual ~StreamReader();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  //! General functions for stream reading and processing

  void read_stream();

  void process_stream();

  int find_string( unsigned char * buffer, int buffer_size, string str ) const;

  int find_header( unsigned char * buffer, int buffer_size, 
                   string & header_name );

  int seek_header( unsigned char * buffer, string & header_name, int & nread );

  int update_input( unsigned char * buffer );

  int remove_headers( unsigned char * old_buffer, unsigned char * new_buffer,
                      int num_bytes );

  bool checksum_valid( unsigned char * processing_buffer, int num_bytes,
                       unsigned char checksum );  

  //! Memory management 

  void cleanup();

  //! Thread related functions

  void kill_helper( Thread * helper_thread );

  //! Stream 2 ( all 64000 sensors get a header)

  void read_sensor_2( unsigned char * buffer, int start, ssize_t nread );

  void process_sensor_2( unsigned char * processing_buffer,
                         int processing_size );

  void update_pc_mesh( vector<struct PointCloudValue> new_pc_values,
                       string dn );

  //! Stream 3 ( 25 "poles" of 40 sensors, 1000 total sensors)

  void read_sensor_3( unsigned char * buffer, int start, ssize_t nread );

  void process_sensor_3( unsigned char * processing_buffer,
                         int processing_size );
 
private:

  //! GUI variables
  GuiString hostname_;
  GuiInt port_;
  GuiString file_read_;
  GuiString file_write_;
  GuiInt stop_sr_;

  //! Threads
  Thread * reader_thread_;
  Thread * proc_thread_;

  //! Thread safety
  ConditionVariable     stream_cond_;
  Mutex                 buffer_lock_;

  FieldOPort *ofp_;

  //! Stream variables
  int stream_socket_; // Socket connection to stream
  vector<string> headers_; // Headers to look for
  unsigned char * final_buffer_; // Buffer to contain raw data read from stream
  int final_size_; // Size of final buffer

  //! Mesh data
  PointCloudWrapper * pcw_;  
  
  bool first_run_;
};


// ****************************************************************************
// ************************** Class: ReaderThread *****************************
// ****************************************************************************

//! Thread that continually reads and buffers data from an mp3 stream, looks
//! for headers in the data and sends the data to the processing thread when 
//! these headers are found.
class DDDASSHARE ReaderThread : public Runnable {
public:
  ReaderThread( StreamReader * sr );
  virtual ~ReaderThread();
  virtual void run();

private:
  StreamReader * stream_reader_;
};

/*===========================================================================*/
// 
// ReaderThread
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
// ~ReaderThread
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
// run
//
// Description : This is essentially a callback that gets called when the 
//               reader thread is initialized.  In this case the thread begins
//               reading a stream when it is initialized.
//
// Arguments   : none
//
void ReaderThread::run()
{
  cout << "(ReaderThread::run) I'm running!" << endl;

  // Read the solution points contained in the data from the stream
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
// ProcessorThread
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
// ~ProcessorThread
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
// run
//
// Description : This is essentially a callback that gets called when the 
//               reader thread is initialized.  In this case the thread begins
//               waiting for available stream data to process as soon as it
//               is initialized.
//
// Arguments   : none
//
void ProcessorThread::run()
{
  cout << "(ProcessorThread::run) I'm running!" << endl;

  // Read and parse the solution points contained in the data from the stream
  stream_reader_->process_stream();
} 

 
DECLARE_MAKER(StreamReader)

/*===========================================================================*/
// 
// StreamReader
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
  cout << "(StreamReader::StreamReader) Inside" << endl;
  
  first_run_ = 1;
}


/*===========================================================================*/
// 
// ~StreamReader
//
// Description : Destructor
//
// Arguments   : none
//
StreamReader::~StreamReader()
{
  cleanup();
}

/*===========================================================================*/
// 
// execute 
//
// Description : The execute function for this module.  This is the control
//               center for the module.
//
// Arguments   : none
//
void StreamReader::execute()
{
  cout << "(StreamReader::execute) Inside" << endl;

  // If this isn't the first run of the module, clean up previous memory
  if( !first_run_ )
  {
    cleanup();
  }

  first_run_ = 0;
  
  // Proceed to do normal stuff assuming that we're starting from scratch with
  // no memory allocated and no variables set

  // Declare output field
  ofp_ = (FieldOPort *)get_oport("Output Sample Field");
  if (!ofp_) {
    error("(StreamReader::execute) Unable to initialize oport 'Output Sample Field'.");
    cerr << "(StreamReader::execute) ERROR: Unable to initialize oport 'Output Sample Field'." << endl;
    return;
  }

  // Do some initializations of member variables
  final_buffer_ = 0;
  final_size_ = 0;
  pcw_ = 0;
  stream_socket_ = -1;

  // Assign the headers (also called sync values) that we want to look for in 
  // the stream
  // ==> Modify this to look for the header of interest for whatever streaming
  // ==> data we're using.

  headers_.push_back( "DDDAS-KTU2" );
  headers_.push_back( "DDDAS-KTU3" );

#ifdef THREAD_ENABLED

  // Create two threads, one that reads and caches away the data.
  // Another that processes stream data and sends it downstream.
  // These threads run as soon as they're declared.

  Runnable * r = new ReaderThread( this );
  reader_thread_ =  new Thread( r, "reader" );

  Runnable * p = new ProcessorThread( this );
  proc_thread_ =  new Thread( p, "processor" );

#else

  // Threads aren't enabled, run code serially
  read_stream();

#endif

}

/*===========================================================================*/
// 
// tcl_command
//
// Description : The tcl_command function for this module.
//
// Arguments   :
//
// GuiArgs& args - GUI arguments
//
// void* userdata - ???
// 
void StreamReader::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

/*===========================================================================*/
// 
// read_stream 
//
// Description : Continually read from the mp3 stream, remove mp3 headers, 
//               and check for recognized headers.  When the first recognized
//               header is found, call the appropriate function to read
//               that data type.  When the reading funtion returns, update
//               the condition variable to wake up the processing thread so 
//               that it will check the data. Note that the buffer variable 
//               must be locked and unlocked so that reads and writes don't 
//               conflict.
// 
// Arguments   : none
//
void StreamReader::read_stream() 
{
  // Get stream socket
  SocketConnection sock_conn;
  stream_socket_ = sock_conn.get_stream( hostname_.get(), port_.get(),
                                         file_read_.get() );

  if( stream_socket_ == -1 )
  {
    error( "(StreamReader::read_stream) Failed to connect to stream" );
    cerr << "(StreamReader::read_stream) ERROR: Failed to connect to stream" << endl;

#ifdef THREAD_ENABLED

    // Make sure processing thread stops
    stop_sr_.set( 1 );
 
    cout << "(StreamReader::read_stream) Broadcasting condition signal" << endl;
    stream_cond_.conditionBroadcast();

    cout << "(StreamReader::read_stream) Unlocking mutex" << endl;
    buffer_lock_.unlock();

#endif

    return;
  }

  int cnt = 0;
  cout << "(StreamReader::read_stream) Receiving data....." << endl;

  // Initialize buffer to store input
  unsigned char buffer[BUFFER_SIZE];

  // Continually read full datasets until told to stop
  stop_sr_.reset();
  while( !stop_sr_.get() )
  {

#ifdef THREAD_ENABLED

    cout << "(StreamReader::read_stream) Locking mutex" << endl;

    // Lock buffer mutex
    buffer_lock_.lock();

#endif

    // Continually read small chunks off of the stream until the first header
    // is found

    string header_name = "NONE";
    int nread = 0;

    // Read until an appropriate header is found
    int start = seek_header( buffer, header_name, nread );

    if( start < 0 )
    {
      cerr << "(StreamReader::read_stream) WARNING: Failed to find "
           << "appropriate header" << endl;

#ifdef THREAD_ENABLED

      // Make sure processing thread stops
      stop_sr_.set( 1 );

      cout << "(StreamReader::read_stream) Broadcasting condition signal" << endl;
      stream_cond_.conditionBroadcast();

      cout << "(StreamReader::read_stream) Unlocking mutex" << endl;
      buffer_lock_.unlock();

#endif

      return; 
    }

    // Call appropriate read function for this type of data
    // ==> Modify this to provide a reading/processing function for whatever 
    // ==> data follows the header that was found.
    if( header_name == "DDDAS-KTU2" )
    {
      cout << "(StreamReader::read_stream) Found DDDAS-KTU2 header" << endl;
      read_sensor_2( buffer, start, nread );
    }
    else if( header_name == "DDDAS-KTU3" )
    {
      cout << "(StreamReader::read_stream) Found DDDAS-KTU3 header" << endl;
      read_sensor_3( buffer, start, nread );
    }
    else
    {
      char error_string[MAX_ERROR_STRING];
      sprintf( error_string, 
               "(StreamReader::read_stream) Unknown header: %s",
               header_name.c_str() );
      error( error_string );
      cerr << "(StreamReader::read_stream) ERROR: Unknown header: " 
           << header_name << endl;
    }

#ifdef THREAD_ENABLED

    // Signal to processing thread that buffer is available
    cout << "(StreamReader::read_stream) Broadcasting condition signal" << endl;
    stream_cond_.conditionBroadcast();

    // Unlock buffer mutex
    //cout << "(StreamReader::read_stream) Unlocking mutex" << endl;
    buffer_lock_.unlock();

    // This is a temporary hack to avoid livelock until I fix the threading
    // bug
    sleep( 1 ); 

#else

    // Call the processing function explicitly
    process_stream();

    // Deallocate memory (should have already been done, but check anyway)
    if( final_buffer_ != 0 )
    {
      delete [] final_buffer_;
      final_buffer_ = 0;
    }
 
#endif

    cnt++;
    
    // Update the stop variable
    stop_sr_.reset();
    
  }	

  // If the module was manually stopped, make sure to signal the processing
  // thread before returning so it doesn't get stuck. Also free any dynamically
  // allocated memory still hanging around
  if( stop_sr_.get() )
  {
    // Deallocate memory
    if( final_buffer_ != 0 )
    {
      delete [] final_buffer_;
      final_buffer_ = 0;
    }

#ifdef THREAD_ENABLED

    // Send signal etc.    
    cout << "(StreamReader::read_stream) Broadcasting condition signal" << endl;
    stream_cond_.conditionBroadcast();

    cout << "(StreamReader::read_stream) Unlocking mutex" << endl;
    buffer_lock_.unlock();

#endif

  } 

  close( stream_socket_ );
  cout << "(StreamReader::read_stream) Leaving" << endl;  

}

/*===========================================================================*/
//
// process_stream
// 
// Description : Parses the contents of a given buffer of data that 
//               has been read from the stream.  Figures out what kind of data
//               is contained in the buffer and passes the data to the 
//               appropriate function for processing.
//
// Arguments   : none
//
void StreamReader::process_stream()
{
  cout << "(StreamReader::process_stream) Inside" << endl;

#ifdef THREAD_ENABLED

  // Continually wait for the reader to set the condition signal, or for the 
  // user to click the "Stop" button on the UI
  stop_sr_.reset();
  while( !stop_sr_.get() )
  {

    cout << "(StreamReader::process_stream) Waiting for condition signal" 
         << endl;

    // Wait for condition signal from reader.  The condition signal indicates 
    // that the reader has buffered enough data for processing.  wait 
    // automatically locks the mutex once it returns, so I don't need to do 
    // that myself
    stream_cond_.wait(buffer_lock_);

    cout << "(StreamReader::process_stream) Got the hand-off" << endl;

    // We got the hand-off from the reader, so now we begin processing the 
    // buffer data

    // Make sure the final_buffer_ was allocated
    if( final_buffer_ == 0 )
    {
      cerr << "(StreamReader::process_stream) WARNING: Final buffer was not "
           << "allocated" << endl;

      // If we got here, it probably means that this thread grabbed the lock
      // again before the reader thread had a chance to get it.  

      // Unlock buffer mutex
      buffer_lock_.unlock();

      // Since the reader didn't grab any data for us, we sleep so that the 
      // reader can grab the lock.  This is a hack to avoid livelock. 
      sleep( 1 );

      // Signal to reader thread that buffer is available
      //stream_cond_.conditionBroadcast();

      // Check stop button
      stop_sr_.reset();
      continue;
    }

#else

    // Make sure the final_buffer_ was allocated
    if( final_buffer_ == 0 )
    {
      cerr << "(StreamReader::process_stream) WARNING: Final buffer was not "
           << "allocated" << endl;
      return;
    }

#endif


    // Make a copy of the final_buffer_ to pass to the specific processing
    // function.  I've done this so that the final_buffer_ can be given back to
    // the reader thread as quickly as possible, although I haven't implemented
    // the thread code to actually do this yet because of threading bugs I need
    // to fix first
    int processing_size = final_size_;
    unsigned char * processing_buffer = new unsigned char[final_size_];
    memcpy( processing_buffer, final_buffer_, 
            processing_size * sizeof(unsigned char) );

    cout << "(StreamReader::process_stream) Initialized input" << endl;

    // Deallocate final buffer memory
    if( final_buffer_ != 0 ) // Just double check
    {
      cout << "(StreamReader::process_stream) Freeing mem from final_buffer_" << endl;
      delete [] final_buffer_;
      final_buffer_ = 0;
    }

    // Check to see what kind of data to process.  This is indicated by the
    // first string in the buffer.
    string str;
    find_header( processing_buffer, processing_size, str );

    if( str == "DDDAS-KTU2" )
    {
      // This is the second test stream in which all 64000 sensors get a 
      // header
      process_sensor_2( processing_buffer, processing_size );
    }
    else if( str == "DDDAS-KTU3" )
    {
      // This is the third test stream with 25 "poles" of 40 sensors, 1000 
      // total sensors (same spec as sensor 2)
      process_sensor_3( processing_buffer, processing_size );
    }
    else
    {
      // Got unrecognized data, spit out an error message and return 
      cerr << "(StreamReader::process_stream) WARNING: Unrecognized header: '" 
           << str << "'" << endl;
    }

    // Deallocate processing buffer memory
    if( processing_buffer != 0 ) // Just double check
    {
      cout << "(StreamReader::process_stream) Freeing mem from processing_buffer" << endl;
      delete [] processing_buffer;
      processing_buffer = 0;
    }

#ifdef THREAD_ENABLED

    stop_sr_.reset();    

    // Unlock buffer mutex
    buffer_lock_.unlock();
  }

#endif

  cout << "(StreamReader::process_stream) Leaving" << endl;
}


/*===========================================================================*/
// 
// find_string
//
// Description : Reads an input stream until a specified string is encountered 
//               or the input ends.  If the string is encountered, returns the 
//               index of the string in the buffer.  Otherwise, 
//               returns -1.
//
// Arguments   :
//
// unsigned char * buffer - The buffer that is to be searched for the given
//                          string
//
// int buffer_size - Number of characters in the buffer
//
// string str - The string to find within the given input stream 
//
int StreamReader::find_string( unsigned char * buffer, int buffer_size, 
                               string str ) const
{
  cout << "(StreamReader::find_string) Inside" << endl;

  // Check byte by byte for an appropriate header in this chunk of data
  for( int i = 0; i < buffer_size; i++ )
  {
    // If there are enough bytes left in the buffer to possible contain
    // a string of this size, check for a match
    int str_len = str.length();
    if( str_len <= (buffer_size - i + 1) )
    {
      // Create the string of that length in the buffer and compare it to
      // the header
      unsigned char buf_str [str_len + 1];
      memcpy( buf_str, &buffer[i], str_len * sizeof(unsigned char) );
      buf_str[str_len] = '\0'; // null-terminate the string
      string buf_string = (const char *) buf_str;
      if( buf_string == str )
      {
        // We found a matching string
        return i;
      }
    }
  }

  cout << "(StreamReader::find_string) Leaving" << endl;
  
  // The string was not found, return -1
  return -1;
}

/*===========================================================================*/
// 
// find_header
//
// Description : Looks for the occurence of any header in the buffer.  Returns
//               the buffer index of the first header found.  Returns the
//               name of the header by reference.  If no header is found, 
//               returns -1.
//
// Arguments   :
//
// unsigned char * buffer - buffer to be searched for headers
// 
// int buffer_size - size of buffer (in bytes)
// 
// string & header_name - name of header found (or unassigned if no header
//                        is found)
//
int StreamReader::find_header( unsigned char * buffer, int buffer_size, 
                               string & header_name )
{
  // Check byte by byte for an appropriate header in this chunk of data
  for( int i = 0; i < buffer_size; i++ )
  {
    // Loop through all the headers, checking for a match
    int num_headers = (int) headers_.size();
    for( int j = 0; j < num_headers; j++ ) 
    {
      // If there are enough bytes left in the buffer to possible contain
      // a header of this size, check for a match
      int header_len = headers_[j].length();

      if( header_len <= (buffer_size - i + 1) )
      {
        // Create the string of that length in the buffer and compare it to
        // the header
        unsigned char buf_str [header_len + 1];
        memcpy( buf_str, &buffer[i], header_len * sizeof(unsigned char) );
        buf_str[header_len] = '\0'; // null-terminate the string
        string buf_hdr = (const char *) buf_str;

        if( buf_hdr == headers_[j] )
        {
          // We found a matching header
          header_name = headers_[j];
          return i;
        }
      }
    }
  }
  return -1;
}

/*===========================================================================*/
// 
// seek_header 
//
// Description : Reads data from a stream until a string is found that matches 
//               one of the entries in the array of header strings.  Returns
//               the buffer index where the header string begins.  
//
// Arguments   :
//
// unsigned char * buffer - Buffer to store char data from socket reads
//
// string & header_name - This gets assigned the name of the header (if any)
//                        that is found.
//
// int & nread - Returned by reference as the number of bytes read in the last
//               read
//
int StreamReader::seek_header( unsigned char * buffer,  
                               string & header_name, int & nread ) 
{
  int index = -1;

  // Continually read small chunks off of the stream until the first header
  // is found
  stop_sr_.reset();      
  while( !stop_sr_.get() )
  {
    cout << "(StreamReader::seek_header) Searching for appropriate header" 
         << endl; 

    // Read chunk of data from stream
    nread = update_input( buffer );

    // Check to make sure data was successfully read from the stream
    if( nread == 0 ) 
    {
      cout << "(StreamReader::seek_header) No data read from stream" << endl;

      // Update the stop variable
      stop_sr_.reset();      
      continue;
    }
    else if( nread < 0 )
    {
      return -1;
    }

    if( (index = find_header( buffer, nread, header_name )) != -1 )
    {
      return index;      
    }

    cout << "(StreamReader::seek_header) No header found yet" << endl;
    stop_sr_.reset();
  }
  
  // Should only get here if the user manually stops the execution of the
  // module
  return -1;
}

/*===========================================================================*/
// 
// update_input
//
// Description : Reads the next chunk of data off of the mp3 stream, removes 
//               the mp3 headers, and updates the input stream and buffer so 
//               that they contain the updated data.  Returns the number of 
//               bytes remaining in buffer after mp3 headers have been removed.
//               Returns
//               Assumes that buffer is of size BUFFER_SIZE.
//
// Arguments   :
//
// unsigned char * buffer - Character buffer of size BUFFER_SIZE that is used
//                          to buffer data read from the mp3 stream.  Data
//                          in this buffer gets copied to the input string 
//                          stream.
//
int StreamReader::update_input( unsigned char * buffer ) 
{
  
  cout << "(StreamReader::update_input) Inside" << endl; 
  //cout << "(StreamReader::update_input) stream_socket_ = " << stream_socket_ 
  //     << endl; 

  int nread = 0;

  bzero( buffer, BUFFER_SIZE );

  // Read chunck of data from stream
  cout << "(StreamReader::update_input) Attempting to read data from stream" 
       << endl;
  if( (nread = read(stream_socket_, buffer, BUFFER_SIZE)) < 0 )
  {
    error( "(StreamReader::update_input) Read failed" );
    cerr << "(StreamReader::update_input) ERROR: Read failed, error follows:" << endl;  
    perror(NULL);
    //cout << "(StreamReader::update_input) Leaving" << endl; 
    return nread; 
  }
  else if( nread == 0 )
  {
    // End of file
    cerr << "(StreamReader::update_input) WARNING: Read 0 bytes" << endl;
    //cout << "(StreamReader::update_input) Leaving" << endl; 
    return nread;
  }

  cout << "(StreamReader::update_input) Read succeeded, nread = " << nread 
       << endl;

  //cout << "(StreamReader::update_input) Removing headers" << endl; 

  // First check for a "404" error message indicating that we weren't 
  // able to retrieve this file from the stream
  int index = find_string( buffer, BUFFER_SIZE, "404 Not found" );
  if( index >= 0 ) 
  {  
    char error_string[MAX_ERROR_STRING];
    sprintf( error_string, 
        "(StreamReader:::update_input) File '%s' not found on server '%s:%i'", 
        (file_read_.get()).c_str(), (hostname_.get()).c_str(), port_.get() );
	error( error_string ); 
 
    cerr << "(StreamReader::update_input) ERROR: File '" 
         << file_read_.get() << "' not found on server '" << hostname_.get() 
         << ":" << port_.get() << "'" << endl;
    return -1;
  }

  // Remove all mp3 headers, create new buffer without the headers
  unsigned char new_buffer[BUFFER_SIZE];
  nread = remove_headers( buffer, new_buffer, nread );

  // Update buffer
  memcpy( buffer, new_buffer, BUFFER_SIZE * sizeof(unsigned char) );

  //cout << "(StreamReader::update_input) Modifying new input" << endl; 

  assert( nread <= BUFFER_SIZE );

  //cout << "(StreamReader::update_input) Leaving" << endl; 
  return nread;
}

/*===========================================================================*/
// 
// remove_headers
//
// Description : Removes all mp3 headers (which occur every 417 bytes) from a 
//               buffer and puts the remaining bytes in a new buffer.  Returns 
//               the number of valid bytes in the new buffer. Fills the 
//               remaining bytes in old_buffer and new_buffer with zeros.  
//               Leaves the mp3 headers in old_buffer.  Assumes that both 
//               old_buffer and new_buffer are of size BUFFER_SIZE.  Only 
//               puts data into new buffer if it is between two mp3 headers
//               417 bytes apart.
//
// Arguments   :
//
// unsigned char * old_buffer - The original buffer that contains mp3 headers
//                              that need to be removed.
//  
// unsigned char * new_buffer - The new buffer that gets assigned the old 
//                              buffer data minus the mp3 headers
//
// int& num_bytes - Number of bytes to examine in old_buffer. 
//
int StreamReader::remove_headers( unsigned char * old_buffer, 
                                  unsigned char * new_buffer, int num_bytes )
{
  // Grab some extra bytes just in case
  int num = num_bytes;
  num_bytes = 0; 
  int bytes_since_header = 0; 
  bool first_header = 1;

  for( int i = 0; i < num; i++ )
  {
    unsigned char ch = old_buffer[i];

    if( ch == 0xff  ) 
    {
      cout << "(StreamReader::remove_headers) bytes_since_header = " 
           << bytes_since_header << endl;
 
      if( num - i < 4 )
      {
        // There aren't enough bytes left in the buffer to fit a header
        // Don't add the remaining bytes to the new buffer, just return
        cerr << "(StreamReader::remove_headers) WARNING: Not enough bytes " 
             << "left to fit header " << endl;
        break; 
      }
      
      // There are enough bytes left for a header to fit.  Check to see if
      // the next 3 bytes match the header 
      printf( "(StreamReader::remove_headers) header = %x %x %x %x\n", 
              old_buffer[i], old_buffer[i+1], old_buffer[i+2], 
              old_buffer[i+3] );    

      if( old_buffer[i+1] == 0xfb && old_buffer[i+2] == 0x90 && 
           old_buffer[i+3] == 0xc0 )
      {
        // We've found a matching header. Check the number of bytes read since 
        // the last header.  
        if( bytes_since_header == 417)
        {
          // Copy the last 417 bytes from the old buffer into the new buffer
          assert( i >= 417 );
          memcpy( &(new_buffer[num_bytes]), &(old_buffer[i - 417]), 
                  417 * sizeof(unsigned char) );     
          num_bytes += 417;
        }
        else if( !first_header )
        {
          // We found a header but it didn't come in the right place so the 
          // data must have been corrupted since the last header.  Respond 
          // to this by throwing away the remaining data and returning with
          // what we've read so far.
          cerr << "(StreamReader::remove_headers) WARNING: Wrong number of " 
               << "bytes since last header, throwing away remaining data" << endl;
          return num_bytes;
        }
        i += 3;
        bytes_since_header = 0;
        first_header = 0;

      }
      else if( !first_header )
      {
        // We must have found a byte somewhere in the data, simply 
        // keep looking for data 
        bytes_since_header++;
      }

    }
    else if( !first_header )
    {
      // Keep looking for data
      bytes_since_header++;
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

  return num_bytes;
}

/*===========================================================================*/
// 
// checksum_valid
//
// Description : Calculates the checksum on this buffer and compares it to the
//               passed checksum.  If they match, returns true.  Returns
//               false otherwise.  The checksum is the complement of the 
//               one-byte sum of all bytes.
//
// Arguments   :
//
// unsigned char * buffer - Buffer of data to run the checksum on.
// int num_bytes - Number of bytes in the buffer.
// unsigned char checksum - 1-byte checksum to compare against.
//
bool StreamReader::checksum_valid( unsigned char * buffer, int num_bytes,
                                   unsigned char checksum )
{
  // Calculate the sum of all bytes in the data
  unsigned int buf_checksum = 0;
  for( int k = 0; k < num_bytes; k++)
  {
    buf_checksum += (unsigned int) buffer[k];
    //cout << "(StreamReader::checksum_valid) buffer[" << k << "] = " 
    //     << buffer[k] << endl;
  } 
 
  // Take the complement of the checksum
  buf_checksum = ~buf_checksum;  

  // Zero out all but the lowest byte
  buf_checksum = buf_checksum & 0xff;

  // Get lowest byte to use as final checksum
  unsigned char buf[sizeof(unsigned int)];
  memcpy( buf, &buf_checksum, sizeof(unsigned int) );  
  unsigned char final_checksum = buf[0];
  
  // The final_checksum is what should be sent as the one byte checksum 
  // at the end of the data

  // Print out the checksum and final_checksum just to make sure the bit
  // operations were done right.  These two values should be equal.
  //cout << "(StreamReader::checksum_valid) final_checksum = " 
  //     << (unsigned int) final_checksum 
  //     << ", checksum = " << (unsigned int) checksum << endl;

  if( final_checksum == checksum )
  {
    return true;
  }
  return false;
}

/*===========================================================================*/
// 
// cleanup
//
// Description : Clean up memory, close open socket, etc.  
//
// Arguments   :
//
void StreamReader::cleanup()
{
  cout << "(StreamReader::cleanup) Inside" << endl; 

  // Deallocate memory
  if( final_buffer_ != 0 )
  {
    delete [] final_buffer_;
    final_buffer_ = 0;
  }

  if( pcw_ != 0 )
  {
    delete pcw_;
    pcw_ = 0;
  }

  headers_.clear();

  // Close open sockets and file descriptors

  if( stream_socket_ != -1 ) close( stream_socket_ );

#ifdef THREADS_ENABLED

  // Kill threads
  kill_helper( reader_thread_ );
  kill_helper( proc_thread_ );

#endif

  cout << "(StreamReader::cleanup) Leaving" << endl; 
}

/*===========================================================================*/
// 
// kill_helper
//
// Description : Kill a specified helper thread.
//
// Arguments   :
//
// Thread * helper_thread - A pointed to the thread to be killed
//
void StreamReader::kill_helper( Thread * helper_thread )  
{
  cout << "(StreamReader::kill_helper) Inside" << endl;

  // kill the helper thread
  if (helper_thread)
  {
    helper_thread->join();
    helper_thread = 0;
  }

  cout << "(StreamReader::kill_helper) Leaving" << endl;
}


// ****************************************************************************
// ********************************* Stream 2 *********************************
// ****************************************************************************


/*===========================================================================*/
// 
// read_sensor_2
// 
// Description : Puts all of the data for one sensor starting from
//               the beginning of a sensor header 'DDDAS-KTU2' into the 
//               final_buffer_ so that it can then be used by process_sensor_2.
//
// Arguments   :
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
void StreamReader::read_sensor_2( unsigned char * buffer, int start, 
                                  ssize_t nread )
{

  // Check to make sure we read enough bytes to contain at least the minimum
  // sized sensor update
  if( nread - start < 100 )
  {
    cerr << "(StreamReader::read_sensor_2) WARNING: Too few bytes read for a "
         << "sensor update, aborting this read" << endl;
    return;    
  }

  // In a later iteration of this implementation I should parse the header to 
  // get number of solutions so that can be factored into the number of bytes 
  // that need to be copied into the final buffer

  // Make new buffer to fit all of the solution points and header
  int cpy_size = nread - start;
  final_size_ = cpy_size;
  final_buffer_ = new unsigned char[final_size_];

  // Set all values of final buffer to 0
  memset( final_buffer_, 0, final_size_ );

  cout << "(StreamReader::read_sensor_2) final_size = " << final_size_ << endl;
  cout << "(StreamReader::read_sensor_2) start = " << start << endl;

  // Copy the first section of the mesh into the final buffer
  assert( cpy_size > 0 && cpy_size <= final_size_
          && cpy_size <= (BUFFER_SIZE - start) );

  memcpy( final_buffer_, buffer + start, cpy_size * sizeof(unsigned char) );   

}

/*===========================================================================*/
// 
// process_sensor_2
//
// Description : Processing code for second sensor spec.  This is hard-coded
//               to handle only a specific version of the spec.
//     
// Arguments   :
//
// unsigned char * processing_buffer - Buffer of raw data read from stream
//
// int processing_size - size of processing_buffer (number of entries in the
//                       array)
//
void StreamReader::process_sensor_2( unsigned char * processing_buffer,
                                     int processing_size ) 
{
  cout << "(StreamReader::process_sensor_2) Inside" << endl;

  unsigned char sync[11];
  unsigned char id[37];
  unsigned char timestamp[22];
  unsigned char num_values[2];
  unsigned char data_name[9];
  unsigned char data_type[7];
  double data_value;
  unsigned char checksum;

  // Open file for output
  string file_write = file_write_.get();
  FILE * debug_output;
  debug_output = fopen( file_write.c_str(), "a" );
 
  // Extract info (string lengths hard-coded)
  memcpy( &sync, &processing_buffer[0], 10 * sizeof(unsigned char) );
  sync[10] = '\0';

  memcpy( &id, &processing_buffer[11], 36 * sizeof(unsigned char) );
  id[36] = '\0';

  memcpy( &timestamp, &processing_buffer[48], 21 * sizeof(unsigned char) );
  timestamp[21] = '\0';

  memcpy( &num_values, &processing_buffer[70], 1 * sizeof(unsigned char) );
  num_values[1] = '\0';

  memcpy( &data_name, &processing_buffer[72], 8 * sizeof(unsigned char) );
  data_name[8] = '\0';

  memcpy( &data_type, &processing_buffer[81], 6 * sizeof(unsigned char) );
  data_type[6] = '\0';

  memcpy( &data_value, &processing_buffer[88], 8 * sizeof(unsigned char) );

  memcpy( &checksum, &processing_buffer[96], sizeof(unsigned char) );

  // Do format conversions to get data into the right format
  int num_vals = atoi( (const char *) num_values );
  string dn = (const char *) data_name;
  string id_str = (const char *) id;

  // Do format conversions to get 3d coordinates and timestamp
  float x, y, z;
  char prefix[80];
  sscanf( (const char *) id, "%9s-%f-%f-%f", prefix, &x, &y, &z );  

  double ts = atof( (const char *) timestamp );  

  cout << "(StreamReader::process_sensor_2) Sensor data:\n"
       << "id = " << id << "\n"
       << "x = " << x << "\n"
       << "y = " << y << "\n"
       << "z = " << z << "\n"
       << "timestamp = " << ts << "\n"
       << "num_vals = " << num_vals << endl;
    
  // Check to see if num_vals is garbage
  if( num_vals > MAX_NUM_VALS )
  {
    cerr << "(StreamReader::process_sensor_2) "
         << "WARNING: num_vals possibly corrupted, num_vals = " << num_vals 
         << ", returning" << endl;
    fprintf( debug_output, "NUM_VALS_CORRUPT\n%i\n\n", num_vals );
    //cout << "NUM_VALS_CORRUPT\n" << num_vals << "\n\n";
    fclose( debug_output );
    return;
  }

  // Check to see if the id prefix is garbage
  if( strcmp(prefix, "VirTelem2") != 0 )
  {
    cerr << "(StreamReader::process_sensor_2) "
         << "WARNING: ID possibly corrupted, prefix = " << prefix 
         << ", returning" << endl;
    fprintf( debug_output, "ID_PREFIX_CORRUPT\n%s\n\n", prefix );
    //cout << "ID_PREFIX_CORRUPT\n" << prefix << "\n" << endl;
    fclose( debug_output );
    return;
  }

  // Set up a queue of values to be added to the point cloud if the CHECKSUM
  // succeeds
  vector<struct PointCloudValue> new_pc_values;

  if( strcmp((const char *) data_type, "scalar") == 0 || 
      strcmp((const char *) data_type, "Scalar") == 0 )
  {
    cout << "(StreamReader::process_sensor_2) Got scalar data" << endl;  

    cout << "(StreamReader::process_sensor_2) data_value " << data_value 
         << endl;
     
    fprintf( debug_output, "<SYNC>\t\t %s\n", sync );
    fprintf( debug_output, "<ID/LOCATION>\t %s-%f-%f-%f\n", prefix, x, y, z );
    fprintf( debug_output, "<TIMESTAMP>\t %20.19f\n", ts );
    fprintf( debug_output, "<NUMVALS>\t %i\n", num_vals );
    fprintf( debug_output, "<NAME>\t\t %s\n", data_name );
    fprintf( debug_output, "<TYPE>\t\t %s\n", data_type );
    fprintf( debug_output, "<DATA>\t\t %f\n", data_value );  
    fprintf( debug_output, "\n" );  

    cout << "<SYNC> " << sync << endl;
    cout << "<ID/LOCATION> " << prefix << "-" << x << "-" 
         << y << "-" << z << endl;
    cout << "<TIMESTAMP> " << ts << endl;
    cout << "<NUMVALS> " << num_vals << endl;
    cout << "<NAME> " << data_name << endl;
    cout << "<TYPE> " << data_type << endl;
    cout << "<DATA> " << data_value << endl;
    cout << endl;

    cout << "(StreamReader::process_sensor_2) Adding data to mesh" << endl;

    // Add this data to a queue of data to be added to the point cloud
    // mesh if/when the the CHECKSUM check succeeds
    struct PointCloudValue pcv;
    Point pt( x, y, z );
    pcv.id = id_str;
    pcv.pt = pt; 
    pcv.data = data_value;
    pcv.data_name = dn;
    new_pc_values.push_back( pcv );      
    cout << "(StreamReader::process_sensor_2) Finished adding data to mesh" << endl;
  }
  else
  {
    cerr << "(StreamReader::process_sensor_2) WARNING: Unrecognized data type "
         << "'" << data_type << "'" << endl; 
    cout << "(StreamReader::process_sensor_2) Processing numvals = " 
         << num_vals <<  endl;       
    fprintf( debug_output, "DATA_TYPE_CORRUPT\n%s\n\n", data_type );
    //cout << "DATA_TYPE_CORRUPT\n" << data_type << "\n" << endl;
    fclose( debug_output );
    return;
  }

  // Check the checksum 
  // The checksum is the compliment of the one-byte sum of all bytes
  //cout << "(StreamReader::process_sensor_2) checksum = " << checksum << endl;

  if( !checksum_valid( processing_buffer, 96, checksum) )
  {
    cerr << "(StreamReader::process_sensor_2) WARNING: Checksum failed,"
         << "discarding data" << endl;
    return;
  }

  fclose( debug_output );

  // Update the point cloud mesh with the new values and send it downstream
  update_pc_mesh( new_pc_values, dn );

  cout << "(StreamReader::process_sensor_2) Leaving" << endl;
}


/*===========================================================================*/
// 
// update_pc_mesh
//
// Description : Update the point cloud mesh with the new values and send it 
//               downstream.
//     
// Arguments   :
//
// vector<struct PointCloudValue> new_pc_values - Vector of values to add
// to the point cloud mesh
//
void StreamReader::update_pc_mesh( vector<struct PointCloudValue> 
                                   new_pc_values, string dn )
{
  // Update the point cloud mesh with the new values
  if( pcw_ == 0 )
  {
    cout << "(StreamReader::update_pc_mesh) Allocating point cloud mesh" << endl;
    pcw_ = new PointCloudWrapper();
  }

  cout << "(StreamReader::update_pc_mesh) Updating point cloud mesh" << endl;

  int npv_size = (int) new_pc_values.size();
  for( int j = 0; j < npv_size; j++ ) 
  {
    struct PointCloudValue new_pcv = new_pc_values[j]; 
    pcw_->update_node_value( new_pcv.id, new_pcv.pt, new_pcv.data, 
                             new_pcv.data_name );
  } 

  cout << "(StreamReader::update_pc_mesh) Freezing point cloud" << endl;

  pcw_->freeze( dn );


  cout << "(StreamReader::update_pc_mesh) Getting point cloud field" << endl;

  // Get the field handle
  PCField pc_fld = pcw_->get_field(dn);
  FieldHandle fld( pc_fld.get_rep() );

  cout << "(StreamReader::update_pc_mesh) Sending point cloud field downstream" << endl;

  if( fld.get_rep() == 0 )
  {
    error( 
      "(StreamReader::update_pc_mesh) Point Cloud Field is NULL"  );
    cerr << "(StreamReader::update_pc_mesh) ERROR: "
         << "Point Cloud Field is NULL" << endl;
    cout << "(StreamReader::update_pc_mesh) Leaving" << endl;
    return;
  }

  // Make a copy of the field
  fld.detach();  
  fld->mesh_detach();

  // Send point cloud mesh downstream
  ofp_->send_intermediate( fld );

}


// ****************************************************************************
// ********************************* Stream 3 *********************************
// ****************************************************************************


/*===========================================================================*/
// 
// read_sensor_3
// 
// Description : Puts all of the data for one sensor starting from
//               the beginning of a sensor header 'DDDAS-KTU3' into the 
//               final_buffer_ so that it can then be used by process_sensor_3. 
//
// Arguments   :
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
void StreamReader::read_sensor_3( unsigned char * buffer, int start, 
                                  ssize_t nread )
{
  // Not implemented
}

/*===========================================================================*/
// 
// process_sensor_3
//
// Description : Processing code for third sensor spec.  
//     
// Arguments   :
//
// unsigned char * processing_buffer - Buffer of raw data read from stream
//
// int processing_size - size of processing_buffer (number of entries in the
//                       array)
//
void StreamReader::process_sensor_3( unsigned char * processing_buffer,
                                    int processing_size ) 
{
  cout << "(StreamReader::process_sensor_3) Inside" << endl;

  // This stream spec has not yet been implemented and tested.  Here is some
  // code that was previously used for sensor 2 and could be used as a starting
  // point for sensor 3.  I make no guarantees about the correctness of this 
  // code.

  /*
  // Convert buffer to input stream since a stream is easier to parse
  string str = (const char *) processing_buffer;
  istringstream input( str, istringstream::in );

  //char test[20];
  //memcpy( test, processing_buffer, sizeof(char) * 20 );
  //printf( "(StreamReader::process_sensor_3) test = '%s'\n", test );

  // Parse the data information the precedeces the actual data values
  string id;
  string timestamp;
  int num_vals;
  string data_name;
  string data_type;
  string header;

  input >> header >> id >> timestamp >> num_vals;

  // Parse id tp get the unique x,y,z coordinates

  // Do format conversions to get 3d coordinates and timestamp
  char id_c_str[80]; 
  float x, y, z;
  char prefix[80];
  strcpy( id_c_str, id.c_str() );
  sscanf( id_c_str, "%9s-%f-%f-%f", prefix, &x, &y, &z );  

  double ts = atof( timestamp.c_str() );  

  cout << "(StreamReader::process_sensor_3) Sensor data:" << endl
       << "id = " << id << "\n"
       << "x = " << x << "\n"
       << "y = " << y << "\n"
       << "z = " << z << "\n"
       << "timestamp = " << ts << "\n"
       << "num_vals = " << num_vals << endl;
    
  // Debugging code
  // Open file for output
  string file_write = file_write_.get();
  FILE * debug_output;
  debug_output = fopen( file_write.c_str(), "a" );
 

  // Check to see if num_vals is garbage
  if( num_vals > MAX_NUM_VALS )
  {
    cerr << "(StreamReader::process_sensor_3) "
         << "ERROR: num_vals possibly corrupted, num_vals = " << num_vals 
         << endl;
    return;
  }

  // Check to see if the id prefix is garbage
  if( strcmp(prefix, "VirTelem2") != 0 )
  {
    cerr << "(StreamReader::process_sensor_3) "
         << "ERROR: ID possibly corrupted, prefix = " << prefix 
         << endl;
    return;
  }

  // Set up a queue of values to be added to the point cloud if the CHECKSUM
  // succeeds
  vector<struct PointCloudValue> new_pc_values;

  int data_offset = 0;

  // Process each "value" sent.  This value can be of type scalar, vector, or
  // tensor
  for( int j = 0; j < num_vals; j++ )
  {
    int pos = input.tellg();

    // Get the data_name and data_type for this value
    input >> data_name >> data_type;

    if( data_type == "scalar" || data_type == "Scalar" )
    {
      cout << "(StreamReader::process_sensor_3) Got scalar data" << endl;  

      // Check to make sure we have enough bytes left in the buffer to contain 
      //this data
      if( processing_size * sizeof(unsigned char) - pos < 30 )
      {
        cerr << "(StreamReader::process_sensor_3) WARNING: Too few bytes to "
             << "process scalar data" << endl;
        return;
      }

      // Get the binary double 
      data_offset = input.tellg();
      double data;

      assert( sizeof(double) <= 
              (processing_size * sizeof(unsigned char) - data_offset + 1) );
      memcpy( &data, &(processing_buffer[data_offset + 1]), sizeof(double) );
      data_offset += 4;

      cout << "(StreamReader::process_sensor_3) data " << data << endl;
     
      fprintf( debug_output, "<SYNC>\t\t DDDAS-KTU2\n" );
      fprintf( debug_output, "<ID/LOCATION>\t %s-%f-%f-%f\n", prefix, x, y, z );
      fprintf( debug_output, "<TIMESTAMP>\t %20.19f\n", ts );
      fprintf( debug_output, "<NUMVALS>\t %i\n", num_vals );
      fprintf( debug_output, "<NAME>\t\t %s\n", data_name.c_str() );
      fprintf( debug_output, "<TYPE>\t\t %s\n", data_type.c_str() );
      fprintf( debug_output, "<DATA>\t\t %f\n", data );  
      fprintf( debug_output, "\n" );  

      // Add this data to a queue of data to be added to the point cloud
      // mesh if/when the the CHECKSUM check succeeds
      struct PointCloudValue pcv;
      Point pt( x, y, z );
      pcv.pt = pt; 
      pcv.data = data;
      pcv.data_name = data_name;
      new_pc_values.push_back( pcv );      
    }
    else if( data_type == "vector" || data_type == "Vector" )
    {
      cout << "(StreamReader::process_sensor_3) Got vector data" << endl;  

      // Check to make sure we have enough bytes left in the buffer to contain 
      //this data
      if( processing_size * sizeof(unsigned char) - pos < 50 )
      {
        cerr << "(StreamReader::process_sensor_3) WARNING: Too few bytes to "
             << "process vector data" << endl;
        return;
      }
    }
    else if( data_type == "tensor" || data_type == "Tensor" )
    {
      cout << "(StreamReader::process_sensor_3) Got tensor data" << endl;  
      // Check to make sure we have enough bytes left in the buffer to contain 
      //this data
      if( processing_size * sizeof(unsigned char) - pos < 80 )
      {
        cerr << "(StreamReader::process_sensor_3) WARNING: Too few bytes to "
             << "process tensor data" << endl;
        return;
      }
    }
    else
    {
      cerr << "(StreamReader::process_sensor_3) ERROR: Unrecognized data type "
           << "'" << data_type << "'" << endl; 
      cout << "(StreamReader::process_sensor_3) Processing numvals = " 
           << num_vals <<  endl;       
      fprintf( debug_output, "DATA_TYPE_CORRUPT\n%s\n\n", data_type.c_str() );
    }
  }

  // Grab the checksum and check it
  // The checksum is the compliment of the one-byte sum of all bytes
  unsigned char checksum;

  assert( sizeof(unsigned char) <= 
          (processing_size * sizeof(unsigned char) - data_offset + 1) );
  memcpy( &checksum, &(processing_buffer[data_offset + 1]), sizeof(unsigned char) );

  cout << "(StreamReader::process_sensor_3) checksum = " << checksum << endl;

  if( !checksum_valid( processing_buffer, data_offset + 1, checksum) )
  {
    cerr << "(StreamReader::process_sensor_3) WARNING: Checksum failed,"
         << "discarding data" << endl;
    return;
  }

  // Update the point cloud mesh with the new values
  if( pcw_ == 0 )
  {
    pcw_ = new PointCloudWrapper();
  }

  int npv_size = (int) new_pc_values.size();
  for( int j = 0; j < npv_size; j++ ) 
  {
    struct PointCloudValue new_pcv = new_pc_values[j]; 
    pcw_->update_node_value( id, new_pcv.pt, new_pcv.data, 
                             new_pcv.data_name );
  } 
  pcw_->freeze( data_name );

  // Get the field
  // Get the field handle
  PCField pc_fld = pcw_->get_field(dn);
  FieldHandle fld( pc_fld.get_rep() );

  cout << "(StreamReader::process_sensor_3) Sending point cloud field downstream" << endl;

  if( fld.get_rep() == 0 )
  {
    cerr << "(StreamReader::process_sensor_3) ERROR: "
         << "Point Cloud Field is NULL" << endl;
    cout << "(StreamReader::process_sensor_3) Leaving" << endl;
    fclose( debug_output );
    return;
  }

  // Make a copy of the field
  fld.detach();  
  fld->mesh_detach();

  // Send point cloud mesh downstream
  ofp_->send_intermediate(fld);

  fclose( debug_output );
  */

  cout << "(StreamReader::process_sensor_3) Leaving" << endl;
}

} // End namespace DDDAS




