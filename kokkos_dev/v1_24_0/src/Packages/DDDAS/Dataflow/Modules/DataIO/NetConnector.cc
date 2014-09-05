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
 * C++ (CC) FILE : NetConnector.cc
 *
 * DESCRIPTION   : This is designed to test and demonstrate the use of
 *                 the new SCIRun socket library comprised of the
 *                 src/Core/Util/Net* files.  This module should allow
 *                 SCIRun to form multiple socket connections to multiple
 *                 external clients and exchange data.
 *
 * AUTHOR(S)     : Jenny Simpson
 *                 SCI Institute
 *                 University of Utah 
 *                 
 * CREATED       : Mon Apr 12 14:25:41 MDT 2004
 * MODIFIED      : Mon Apr 12 14:25:41 MDT 2004
 * NOTES         : 
 *
 *  Copyright (C) 2003 SCI Group
 */

// SCIRun includes

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <Packages/DDDAS/share/share.h>
#include <Core/Util/Comm/NetInterface.h>
//#include <Packages/DDDAS/Core/Utils/NetInterface.h>
 
// Standard lib includes

#include <iostream>
#include <fstream>
#include <assert.h>
#include <sys/types.h>

#define SERVER 0
#define CLIENT 1

namespace DDDAS {

using namespace SCIRun;
  
// ****************************************************************************
// **************************** Class: NetConnector ***************************
// ****************************************************************************

// Helper functions
void read_callback( NetInterface * interface, NetConnection * conn, 
                    char * recv_buffer, int num_read );
void listen_callback( NetInterface * interface, NetConnection * conn );

class DDDASSHARE NetConnector : public Module {

public:

  //! Virtual interface

  // !Constructors
  NetConnector(GuiContext* ctx);

  // !Destructor
  virtual ~NetConnector();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  void stop_interface();

private:

  //! Member variables
  NetInterface interface_;

  //! GUI variables
  GuiString cliserv_;
  GuiInt test_;
  GuiInt stop_;
};
 
 
DECLARE_MAKER(NetConnector)

/*===========================================================================*/
// 
// NetConnector
//
// Description : Constructor
//
// Arguments   : 
//
// GuiContext* ctx - GUI context
//
NetConnector::NetConnector(GuiContext* ctx)
  : Module("NetConnector", ctx, Source, "DataIO", "DDDAS"), 
    cliserv_(ctx->subVar("cliserv")),
    test_(ctx->subVar("test")),
    stop_(ctx->subVar("stop"))
{ 
}
 
/*===========================================================================*/
// 
// ~NetConnector
//
// Description : Destructor
//
// Arguments   : none
//
NetConnector::~NetConnector()
{
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
void NetConnector::execute()
{

  cliserv_.reset();
  string type = cliserv_.get();

  cout << "(NetConnector::execute) Running " << type << endl;

  if( type == "Server" )
  {
    //
    // Server code
    //

    // Get which test to run
    test_.reset();
    int test_num = test_.get();

    cout << "(NetConnector::execute) Running test number " <<  test_num 
         << endl;

    // Listen for incoming connection requests
    interface_.listen( 9999, 0, listen_callback );

    // Print existing connections
    interface_.print_connections();

    switch( test_num )
    {
      case 1 :
        // Do nothing
        break;
      case 2 :
        sleep( 10 );

        // Disconnect from all current connections
        interface_.disconnect_all();
        break;
      case 3 :
        sleep( 10 );

        // Stop listening
        cout << "(NetConnector::execute) Stopping listener" << endl;
        interface_.stop_listen();
        cout << "(NetConnector::execute) Listener stopped" << endl;
        //while( 1 );
        break;
      default :
        cout << "(NetConnector::execute) Unknown test number " << test_num 
             << endl;
        break;
    }

    // Print existing connections
    interface_.print_connections();

    //cout << "(NetConnector::execute) Waiting for listener" << endl;

    // Wait for listening thread to die (blocking call)
    //interface_.wait_listener();

  }
  else
  {
    //
    // Client code
    //

    // Get which test to run
    test_.reset();
    int test_num = test_.get();

    cout << "(NetConnector::execute) Running test number " << test_num << endl;

    // Connect to a remote host
    NetConnection conn = interface_.connect( "127.0.0.1", 9999 );

    if( !conn.is_open() )
    {
      cerr << "(NetConnector::execute) ERROR: Connection attempt failed\n" 
           << endl;
      return;
    }

    cout << "(NetConnector::execute) Printing connections" << endl;
    interface_.print_connections();
    conn.print();

    char recv_buff[MAXLINE+1];
    int num_read;

    switch( test_num )
    {
      case 1 :

        // Start a reader that continually reads from the connection
        cout << "(NetConnector::execute) Running read loop" << endl;
        interface_.start_read_loop( &conn, read_callback );
        //interface_.wait_reader( &conn );
        break;
      case 2 :
        cout << "(NetConnector::execute) Before read" << endl;  
        num_read = interface_.read( &conn, recv_buff, MAXLINE );
        recv_buff[num_read] = '\0';
        cout << "(NetConnector::execute) Data read: " << recv_buff << endl; 
        interface_.disconnect( &conn );
        cout << "(NetConnector::execute) After disconnect" << endl;
        interface_.print_connections();
        break;
      case 3 :
        cout << "(NetConnector::execute) Before read" << endl;  
        num_read = interface_.readline( &conn, recv_buff, MAXLINE );
        recv_buff[num_read] = '\0';
        cout << "(NetConnector::execute) Data read: " << recv_buff << endl; 
        interface_.disconnect( &conn );
        cout << "(NetConnector::execute) After disconnect" << endl;
        interface_.print_connections();
        break;
      case 4 :
        interface_.disconnect( &conn ); 
        cout << "(NetConnector::execute) After disconnect" << endl;
        interface_.print_connections();
        break;      
      case 5 :
        // Start a reader that continually reads from the connection
        cout << "(NetConnector::execute) Running read loop" << endl;
        interface_.start_read_loop( &conn, read_callback );
        sleep( 10 );
        cout << "(NetConnector::execute) Stopping read loop" << endl;
        interface_.stop_read_loop( &conn );
        cout << "(NetConnector::execute) Read loop stopped" << endl;
        break;
      case 6 :
        cout << "(NetConnector::execute) Before read" << endl;  
        num_read = interface_.readn( &conn, recv_buff, 5 );
        recv_buff[num_read] = '\0';
        cout << "(NetConnector::execute) Data read: " << recv_buff << endl; 
        interface_.disconnect( &conn );
        cout << "(NetConnector::execute) After disconnect" << endl;
        interface_.print_connections();
        break;      
      case 7 :
        cout << "(NetConnector::execute) Before read" << endl;  
        num_read = interface_.recv( &conn, recv_buff, MAXLINE, 0 );
        recv_buff[num_read] = '\0';
        cout << "(NetConnector::execute) Data read: " << recv_buff << endl; 
        interface_.disconnect( &conn );
        cout << "(NetConnector::execute) After disconnect" << endl;
        interface_.print_connections();
        break;      
      default :
        cout << "(NetConnector::execute) Unknown test number " << test_num 
             << endl;
        return;
    }
  }

  cout << "(NetConnector::execute) Checking stop var" << endl;
  while( !stop_.get() ) stop_.reset();
  interface_.stop();
  cout << "(NetConnector::execute) Returning" << endl;
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
void NetConnector::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

/*===========================================================================*/
//
// stop_interface
//
// Description : Stop the Network interface
//
// Arguments   : none
//
void NetConnector::stop_interface()
{
  interface_.stop();
}

/*===========================================================================*/
//  
// read_callback
// 
// Description : Callback function that processes data read from a connected
//               socket.
//
// Arguments   : 
//
// NetConnection * conn - Connection data was read from.
//
// char * recv_buffer - Buffer containing data read.
//
// int num_read - Number of bytes read from connection.
//
void read_callback( NetInterface * interface, NetConnection * conn, 
                    char * recv_buffer, int num_read )
{
  cout << "(read_callback) Inside" << endl;
  char buff_str[num_read + 1];
  strncpy( buff_str, recv_buffer, num_read );
  buff_str[num_read] = '\0';
  
  cout << "(read_callback) Data read: " << buff_str << endl;

  // Open file for output
  FILE * output;
  output = fopen( "output_file.txt", "a" );
  
  fwrite( recv_buffer, sizeof(char), num_read, output );  

  fclose( output );

  //interface->writen( conn, recv_buffer, num_rtead ); 

} 

/*===========================================================================*/
//  
// listen_callback
// 
// Description : Callback function that processes each new connection
//               accepted by the listener.
//
// Arguments   : 
//
// struct NetConnectionC * conn - New connection to process.
//
void listen_callback( NetInterface * interface, NetConnection * conn )
{
  cout << "(listen_callback) Inside" << endl;

  // Run a reader for this connection
  interface->start_read_loop( conn, read_callback );

  /*
  // Open file for input
  FILE * input;
  input = fopen( "test_file.txt", "r" );
  
  char ch;
  while( fread(&ch, sizeof(char), 1, input) > 0 && conn->is_open() )
  {
    interface->writen( conn, &ch, 1 );
  }

  fclose( input );
  */

  // Write test data to connection
  while( conn->is_open() )
  {
    cout << "(listen_callback) Writing data" << endl;
    interface->writen( conn, "Hello from server\n", 18 );

    //interface->send_message( conn, PING );
    //interface->writen( conn, "Hello from server\n", 18 );
    //interface->send_message( conn, DISCONNECT );

    // Print existing connections
    //interface->print_connections();

    sleep( 2 );
  }
  
} 

} // End namespace DDDAS




