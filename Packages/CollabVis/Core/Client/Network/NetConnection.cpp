/*
 *
 * NetConnection: Abstraction for a network connection.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: April 2001
 *
 */

#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/times.h>
#include <Network/NetConnection.h>
#include <Network/NetDispatchManager.h>
#include <Network/NetInterface.h>
#include <Malloc/Allocator.h>
#include <Logging/Log.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

namespace SemotusVisum {

#define _USE_DETACH
#define CLK_TCK 100 // TEMPORARY code for compilation purposes -- remove me!

using namespace SCIRun;

NetConnectionReader::NetConnectionReader( NetConnection *parent ) :
  dieNow( false ), done( false ), parent( parent ) {
}

NetConnectionReader::~NetConnectionReader() {
}

void
NetConnectionReader::run() {
  Log::log( ENTER, "[NetConnectionReader::run] entered" );
  
  // Do a check to see which transfer mode we are using

  // If the transfer mode is IP Multicast, use alternate code

   cerr << "In NetConnectionReader::run, thread id = " << pthread_self() << endl;
  /* Wait for data to be read on our connection. */
  std::cerr << "NCR is pid " << getpid() << endl;
  parent->readerPID = getpid();

  //SemotusVisum::Thread::Thread::self()->makeInterruptable();
  while ( !dieNow ) {
    
    unsigned dataSize;
    int bytesread = 0;
    dataSize = 0;

    cerr << "NetConnectionReader::run - sizeof(unsigned int) = " << sizeof(unsigned int) << endl;
    cerr << "NetConnectionReader::run - (int)sizeof(unsigned int) = " << (int)sizeof(unsigned int) << endl;
    cerr << "Should be calling appropriate read function now" << endl;

    /* Read the data size from the network. */
    if ( ( bytesread =
	   ( parent->connection.read( (char *)&dataSize,
				      sizeof(unsigned int) ) ) )
	 < (int)sizeof(unsigned int) ) {

      // If we've timed out, continue
      if ( bytesread == -9 )
	continue;
      
      // If we've already dealt with this, just return.
      if ( parent->cleanDisconnect ) {
	done = true;
	Log::log( DEBUG, "[NetConnectionReader::run] Reader exiting1" );
	//return;
	Thread::exit();
      }
    
      Log::log( ERROR, string("[NetConnectionReader::run] Only read ") + mkString(bytesread) + " of " +
		mkString( sizeof( unsigned int ) ) + " bytes!" );

      /* Disconnection code. */
      Log::log( WARNING, string(parent->name) +
		" disconnected uncleanly. Reason: " + strerror(errno) );
      
      parent->cleanDisconnect = false;
      parent->netMonitor.getMailbox().send( parent );
      done = true;
      Log::log( DEBUG, "[NetConnectionReader::run] Reader exiting2" );
      return;
    }

    /* Convert the data */
    DataDescriptor dd = SIMPLE_DATA( UNSIGNED_INT_TYPE, 1 );
    ConvertNetworkToHost( (void *)&dataSize,
			  (void *)&dataSize,
			  &dd,
			  1 );

    Log::log( DEBUG, string("[NetConnectionReader::run] Upcoming data is ") + mkString(dataSize) +
	      " bytes!");

    if ( dataSize == 0 ){
      // continue; // EJL
    }
    /* The client always sends us text */
    char * buffer = scinew char[ dataSize + 1 ];
    memset( buffer, 0, dataSize + 1 );
    
    if ( dieNow ) {
      done = true;
      Log::log( DEBUG, "[NetConnectionReader::run] Reader exiting3" );
      return;
    }
    
    if ( buffer == NULL )
      continue;
    int bytesRead = parent->connection.read( buffer, dataSize );
    Log::log( DEBUG, "[NetConnectionReader::run] Done reading!" );
    
    // Check to make sure that bytesRead corresponds to the number of 
    // bytes expected
    if ( bytesRead != dataSize ){
      Log::log( ERROR, "[NetConnectionReader::run] Incorrect number of bytes read.  bytestRead = " + mkString(bytesRead) + " dataSize = " + mkString(dataSize) );
         
      // take appropriate action
    }

    // If we got an error, and weren't interrupted, we're probably
    // no longer connected...
    if ( bytesRead <= 0 ) {
      delete buffer;
      
      Log::log( WARNING, string("[NetConnectionReader::run] Disconnected. Reason: ") + strerror(errno) );
      
      
      if ( dieNow ) {
	done = true;
	Log::log( DEBUG, "[NetConnectionReader::run] Reader exiting4" );
	return;
      }
      
      parent->netMonitor.getMailbox().send( parent );
      done = true;
      Log::log( DEBUG, "[NetConnectionReader::run] Reader exiting5" );
      return;
    }
    
    Log::log( DEBUG, string("[NetConnectionReader::run] Read ") + mkString(bytesRead) + " bytes" );
    Log::log( DEBUG, buffer );
    
    /* Convert the data - the client always sends us text */
    char * buffer1 = scinew char[ dataSize + 1 ];
    HomogenousConvertNetworkToHost( (void *)buffer1, (void *)buffer,
				    CHAR_TYPE, dataSize + 1 );
    
    // Pass the data to the network dispatch manager.
    if ( dieNow ) {
      done = true;
      Log::log( DEBUG, "[NetConnectionReader::run] Reader exiting6" );
      return;
    }
    if ( NetConnection::useDispatchManager )
      NetDispatchManager::getInstance().fireCallback( buffer1,
						      bytesRead,
						      parent->name );
    else {
      /* Call the callback function if it isn't NULL */
      if ( NetConnection::callbackFunction != NULL ) {
	(*NetConnection::callbackFunction)( (void *)buffer1 );
      }
    }
    delete buffer;
    delete buffer1;
    Log::log( DEBUG, "[NetConnectionReader::run] Restarting read loop" );
  }
  
  Log::log( LEAVE, "[NetConnectionReader::run] leaving, thread id = " + mkString((int) pthread_self()) );
}

NetConnectionWriter::NetConnectionWriter( NetConnection *parent ) :
  dieNow( false ), parent( parent ) {
}

NetConnectionWriter::~NetConnectionWriter() {
}

void
NetConnectionWriter::run() {
  Log::log( ENTER, "[NetConnectionWriter::run] entered, thread id = " + mkString((int) pthread_self()) );
  dataItem d;
  struct tms dummy;
  double start, end;
  double ci = 1./CLK_TCK;
  
  /* Wait for data to be written on our connection. */
  std::cerr << "NCW is pid " << getpid() << endl;

  //SemotusVisum::Thread::Thread::self()->makeInterruptable();
  while ( !dieNow ) {
  
    /* Writing data - wait for messages (data) in our inbox */
    d = parent->mbox.receive();

    start = (double)times(&dummy)*ci;
    
    if ( dieNow ) {
      done = true;
      Log::log( DEBUG, "[NetConnectionWriter::run] Writer exiting" );
      return;
    }
    
    int bytesWritten = 0;
    unsigned int dataSize = d.getSize();
    Log::log( DEBUG, "[NetConnectionWriter::run] Writing message to network:" );
    Log::log( DEBUG, "[NetConnectionWriter::run] " + mkString(d.getData()) );
    
    /* Convert the data size */
    DataDescriptor dd = SIMPLE_DATA( UNSIGNED_INT_TYPE, 1 );
    ConvertHostToNetwork( (void *)&dataSize,
			  (void *)&dataSize,
			  &dd,
			  1 );
			  
    // Write the data size to our connection
    bytesWritten = parent->connection.write( (const char *)&dataSize,
					     sizeof( unsigned int ) );
    //std::cerr << "Wrote " << bytesWritten << " bytes." << endl;
    if ( bytesWritten != sizeof( unsigned int ) ) {

      Log::log( ERROR, string("[NetConnectionWriter::run] Write error - wrote ") +
		mkString(bytesWritten) + "bytes (data size)" );

      // Punt.
      parent->cleanDisconnect = false;
      parent->netMonitor.getMailbox().send( parent );
      done = true;
      Log::log( DEBUG, "[NetConnectionWriter::run] Writer exiting" );
      return;
    }
    //std::cerr << "Writing data to client" << endl;

    /* If the type of the data is -1, we assume that the data has been
       blessed - ie, it is already suitable for network consumption.
       Only if the data type is not == -1 do we convert the data. */
    if ( d.getType() != -1 /*Mandatory*/&& d.getType() != -2 ) {
      

      /*      if ( d.getType() == CHAR_TYPE ) {
	//std::cerr << "Data before: " << d.getData() << endl;
      }
      else 
	std::cerr << "Data is of type " << d.getType() << endl;
      */
      char *buffer = scinew char[ d.getSize() ];
      HomogenousConvertHostToNetwork( (void *)buffer,
				      (void *)(d.getData()),
				      (DataTypes)d.getType(),
				      d.getSize() );
      
      bytesWritten = parent->connection.write( buffer,
					       d.getSize() ); 
      Log::log( DEBUG, "[NetConnectionWriter::run] Wrote DATA1");
      delete buffer;
    }
    else {
      // Write the data to our connection
      bytesWritten = parent->connection.write( d.getData(),
					       d.getSize() );
      Log::log( DEBUG, "[NetConnectionWriter::run] Wrote DATA2");
    }
    //std::cerr << "Done writing" << endl;
    // If there's an error, log it and move on.
    if ( bytesWritten != d.getSize() ) {
      
       Log::log( ERROR, string("[NetConnectionWriter::run] Write error - wrote ") +
		mkString( bytesWritten ) + "/" + mkString(d.getSize()) +
		". Error = " + strerror(errno) );
      
      // Punt.
      parent->cleanDisconnect = false;
      parent->netMonitor.getMailbox().send( parent );
      done = true;
      Log::log( DEBUG, "[NetConnectionWriter::run] Writer exiting" );
      return;
    }

    // Free memory if needed.
    Log::log( DEBUG, "[NetConnectionWriter::run] Purging..." );
    d.purge();
    end = (double)times(&dummy)*ci;
    std::cerr << "Write took " << ( end - start ) << " seconds " << endl;
  }
  Log::log( LEAVE, "[NetConnectionWriter::run] leaving, thread id = " + mkString((int) pthread_self()) );
}



NetMonitor::~NetMonitor() {

}

void
NetMonitor::run() {
  Log::log( ENTER, "[NetMonitor::run] entered, thread id = " + mkString((int) pthread_self()) );
  std::cerr << "NetMonitor is pid " << getpid() << endl;
  NetConnection * removeConnection;
  
  for( ;; ) {
    
    // Do we have connections that need to be disposed of?
    removeConnection = removeBox.receive();

    if ( removeConnection == NULL ) { // Remove ALL connections, PTP and MC
      continue;
    }
    
    Log::log( MESSAGE, "[NetMonitor::run] Removing connection!" );
    

    Log::log( DEBUG, "[NetMonitor::run] Deleted dead connection" );
  }
  Log::log( LEAVE, "[NetMonitor::run] leaving, thread id = " + mkString((int) pthread_self()) );
}

//////////
// Instantiation of incoming Thread.
Thread *
NetConnection::incomingThread = NULL;

//////////
// Instantiation of the list of all active connections.
  //list<NetConnection *>
  //NetConnection::connectionList;

//////////
// Instantiation of connection list lock.
  //CrowdMonitor
  //NetConnection::connectionListLock( "ConnectionListLock" );

//////////
// Instantiation of network monitor.
NetMonitor
NetConnection::netMonitor;

//////////
// Instantiation of dispatch manager usage flag.
bool
NetConnection::useDispatchManager = true;

//////////
// Instantiation of callback function (when dispatch manager is not
// in use.
void
(*NetConnection::callbackFunction)(void *) = NULL;
  
NetConnection::NetConnection( Connection &connection,
			      const string &name, int flags ) :
  name( name ),
  connection(connection),
  mbox( name.data(), MAX_PENDING_ITEMS ), cleanDisconnect( false )
  
{
  Log::log( ENTER, "[NetConnection::NetConnection] entered, thread id = " + mkString((int) pthread_self()) );
  /* Set up incoming data thread */
  if ( incomingThread == NULL ) {
    incomingThread =
      scinew Thread( &NetConnection::netMonitor,
		     "Network Monitor" );
    incomingThread->detach();
  }

  //std::cerr << "Constructor. This = " << (void *)this << endl;
  
  /* Create helpers and threads */
  char * threadName = NULL;
  if ( flags == READ_ONLY || flags == READ_WRITE ) {
    
    // Set the name 
    threadName = scinew char[ name.length() + 2 ];
    snprintf( threadName, name.length() + 1,
	      "%sR", name.data() );
	     
    //std::cerr << "In NetConnection::NetConnection, initializing readThread" << std::endl;
    Reader = scinew NetConnectionReader( this );
    readThread  = scinew Thread( Reader, threadName );
    
#ifdef _USE_DETACH
    //std::cerr << "Before detach, current thread id is " << pthread_self() << std::endl; 
    readThread->detach();
    //std::cerr << "After detach, current thread id is " << pthread_self() << std::endl; 
    readThread = NULL;
#endif
  }
  else {
    Reader = NULL; readThread = NULL;
  }
    
  if ( flags == WRITE_ONLY || flags == READ_WRITE ) {
    
    // Set the name 
    threadName = scinew char[ name.length() + 2 ];
    snprintf( threadName, name.length() + 1, "%sW", name.data() );
	     
    Writer = scinew NetConnectionWriter( this );
    //std::cerr << "In NetConnection::NetConnection, initializing writeThread" << std::endl;
    writeThread = scinew Thread( Writer, threadName );
    
#ifdef _USE_DETACH
    writeThread->detach();
    writeThread = NULL;
#endif
  }
  else {
    Writer = NULL; writeThread = NULL;
  }
  
  /* Add ourselves to the list of network connections. */

  // Lock the list
  // connectionListLock.writeLock();
  
  // Add ourselves
  // connectionList.push_front( this );
  
  // Unlock the list
  // connectionListLock.writeUnlock();
  Log::log( LEAVE, "[NetConnection::NetConnection] leaving, thread id = " + mkString((int) pthread_self()) );
}


NetConnection::NetConnection( const NetConnection& netConnect) :
  
  name( netConnect.name ),
  connection( netConnect.connection ),
  mbox( netConnect.name.data(), netConnect.mbox.size() )
  
{
  Log::log( ENTER, "[NetConnection copy] entered, thread id = " + mkString((int) pthread_self()) );
  // When we copy a connection, we need not add it to the global list.
  std::cerr << "Copy constructor" << endl;
  Log::log( LEAVE, "[NetConnection copy] leaving, thread id = " + mkString((int) pthread_self()) );
}


NetConnection::~NetConnection() {
  Log::log( ENTER, "[NetConnection destructor] entered, thread id = " + mkString((int) pthread_self()) );
  /* Wait until our helpers are done. */
  
  /* If this disconnect was clean, our reader is still running (blocked on
     data read). Kill it, and collect the thread (along with our knees ). */
#ifndef _USE_DETACH
  
  if ( readThread && !cleanDisconnect ) {
    Log::log( Logging::DEBUG, "[NetConnection destructor] Destroying read thread" );
    Reader->dieNow = true;
    //std::cerr << "Before join, current thread id is " << pthread_self() << std::endl; 
    readThread->join();
    //std::cerr << "After join, current thread id is " << pthread_self() << std::endl; 
  }
  
  
  sleep(1); // Let the threads finish up.
#else
  if ( Reader && !cleanDisconnect ) {
    Log::log( DEBUG, "[NetConnection destructor] Destroying read thread" );
    Reader->dieNow = true;
  }
#endif

  Log::log( DEBUG, string("[NetConnection destructor] Joined read thread ") + name  );
  
  if ( Writer ) {
    Writer->dieNow = true;
    dataItem di;
    mbox.send( di );
#ifndef _USE_DETACH
    writeThread->join();
#endif
  }
  Log::log( DEBUG, string("[NetConnection destructor] Joined write thread ") + name );

  delete &connection;

  /* We don't delete name right now for debugging purposes. It makes our
     name garbage to the thread lib. */
  //  delete name;
  //  delete nickname;
  Log::log( DEBUG, "[NetConnection destructor] Done with nc destructor" );
  Log::log( LEAVE, "[NetConnection destructor] leaving, thread id = " + mkString((int) pthread_self()) );
}


bool
NetConnection::operator==( const NetConnection &nc ) {
  Log::log( ENTER, "[NetConnection assignment] entered, thread id = " + mkString((int) pthread_self()) );
  return true;
  /*  std::cerr << "Names: " << !strcmp( name, nc.name ) << endl;
  std::cerr << "Threads: " << (myThread == nc.myThread) << endl;
  cerr << "End of NetConnection assignment, thread id = " << pthread_self() << endl;
  return ( !strcmp( name, nc.name ) && myThread == nc.myThread );*/
}

}
