/*
 *
 * NetDispatchManager: Provides callback registration for network access.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: February 2001
 *
 */

#include <Network/NetDispatchManager.h>
#include <Malloc/Allocator.h>
#include <Network/NetInterface.h>
#include <Util/Timer.h>
#include <Rendering/ImageRenderer.cpp>

namespace SemotusVisum {


// Instantiation of the singleton net dispatch manager
NetDispatchManager*
NetDispatchManager::manager;

// Instantiation of the base connection ID.
int
NetDispatchManager::baseID = 1;

// Client register null mailbox
Mailbox<MessageData>
clientRegister::nullMbox("NULL",0);

int
NetDispatchManager::registerCallback( message_t tag,
				      Mailbox<MessageData>& mbox,
				      bool persistent ) {
  Log::log( ENTER, "[NetDispatchManager::registerCallback1] entered, thread id = " + mkString((int) pthread_self()) );
  Log::log( DEBUG, string("[NetDispatchManager::registerCallback1] Adding callback for message ") +
	    MakeMessage::messageText( tag ) );
  
  clientRegister * cr = scinew clientRegister( tag,
					       mbox,
					       persistent,
					       baseID++ );
  
  // Add the data to the list of registered clients
  registeredClients.push_front( *cr );

  Log::log( LEAVE, "[NetDispatchManager::registerCallback1] leaving, thread id = " + mkString((int) pthread_self()) );
  return baseID - 1;
}

int
NetDispatchManager::registerCallback( message_t tag,
				      callbackFunc function,
				      void * object,
				      bool persistent ) {
  Log::log( ENTER, "[NetDispatchManager::registerCallback2] entered, thread id = " + mkString((int) pthread_self()) );
  Log::log( DEBUG, string("Adding callback for message ") +
	    MakeMessage::messageText( tag ) );
  clientRegister * cr = scinew clientRegister( tag,
					    function,
					    object,
					    persistent,
					    baseID++ );

  // Add the data to the list of registered clients
  registeredClients.push_front( *cr );

  Log::log( LEAVE, "[NetDispatchManager::registerCallback2] leaving, thread id = " + mkString((int) pthread_self()) );
  return baseID - 1;
}

void                       
NetDispatchManager::deleteCallback( const int ID ) {
  Log::log( ENTER, "[NetDispatchManager::deleteCallback] entered, thread id = " + mkString((int) pthread_self()) );
  Log::log( DEBUG, "[NetDispatchManager::deleteCallback] Deleting callback ID: " + mkString(ID) );
  // Iterate through the list looking for the particular ID
  list<clientRegister>::iterator i;
  
  for ( i = registeredClients.begin();
	i != registeredClients.end();
	i++ )
    if ( i->ID == ID ) {
      registeredClients.erase( i );
      return;
    }
  Log::log( LEAVE, "[NetDispatchManager::deleteCallback] leaving, thread id = " + mkString((int) pthread_self()) );
}

void                       
NetDispatchManager::deleteCallback( const message_t type ) {
  Log::log( ENTER, "[NetDispatchManager::deleteCallback] entered, thread id = " + mkString((int) pthread_self()) );
  Log::log( DEBUG, "[NetDispatchManager::deleteCallback] Deleting callbacks of type " +
	    MakeMessage::messageText( type ));
  
  // Iterate through the list looking for the particular type.
  list<clientRegister>::iterator i;
  
  for ( i = registeredClients.begin();
	i != registeredClients.end();
	i++ )
    //if ( !strcmp( i->type, type ) )
    if ( i->type == type )
      registeredClients.erase( i );

  Log::log( LEAVE, "[NetDispatchManager::deleteCallback] leaving, thread id = " + mkString((int) pthread_self()) );
}

void                       
NetDispatchManager::fireCallback( const char * data,
				  const int numBytes,
				  const string client) {
  Log::log( ENTER, "[NetDispatchManager::fireCallback] entered, thread id = " + mkString((int) pthread_self()) );

  PreciseTimer p;
  p.start();
  
  char buffer[ 1000 ];
  snprintf( buffer, 256, "Got %d bytes of data to fire callback. Data:",
	    numBytes );
  Log::log( DEBUG, string("[NetDispatchManager::fireCallback] Got ") + mkString(numBytes) +
	    " bytes of data to fire callback. Data:" );
  Log::log( DEBUG, "[NetDispatchManager::fireCallback] " + mkString(data) );
  //cerr << "Data: " << data << endl;
  
  /* Get valid data */
  char * goodData = findData( data );
  cerr << "Found good data: " <<  endl;
  if ( goodData == NULL ) { 
    // No good data here
    Log::log( ERROR,
	      "[NetDispatchManager::fireCallback] Did not find any valid data from network!" );
    return;
  }

  //cerr << "goodData = " << goodData << endl;

  message_t messageType;
  cerr << "Building message" << endl;
  MessageBase *m = MakeMessage::makeMessage( (void *)goodData, messageType );
  cerr << "Done" << endl;
  
  
  /* Check message */
  if ( m == NULL ) {
    if ( messageType == ERROR_M ) {
      Log::log( ERROR, "[NetDispatchManager::fireCallback] Error in building message!" );
      Log::log( DEBUG, "[NetDispatchManager::fireCallback] " + mkString(goodData) );
    }
    else if ( messageType == UNKNOWN_M ) {
      Log::log( ERROR, "[NetDispatchManager::fireCallback] Unknown message type from server" );
      Log::log( DEBUG, "[NetDispatchManager::fireCallback] " + mkString(goodData) );
    }
    else {
      Log::log( ERROR, string("[NetDispatchManager::fireCallback] BUG - '") +
		MakeMessage::messageText( messageType ) +
		"' message type, but no message!" );
      Log::log( DEBUG, "[NetDispatchManager::fireCallback] " + mkString(goodData) );
    }
    return;
  }
  else {
    Log::log( DEBUG, string("[NetDispatchManager::fireCallback] Got a '") +
	      MakeMessage::messageText( messageType ) +
	      "' message" );
  }

  cerr << "FireCallback: MakeMessage took " << p.time() * 1000.0 << " ms." << endl;
  p.clear();
  vector<clientRegister*> clients = searchCallbackList( messageType );

  if ( clients.size() > 0 ) {
    for ( unsigned q = 0; q < clients.size(); q++ ) {
      bool result=true;
      int extraBytes = numBytes - strlen( goodData ) - 4;
      MessageData *md = NULL;
      if ( extraBytes > 0 ) {
	Log::log( DEBUG, "[NetDispatchManager::fireCallback] Got " + mkString( extraBytes ) + " bytes of data" );
	char * extraData = scinew char[ extraBytes ];
	memcpy( extraData, data + strlen( goodData ) + 4,
		extraBytes );
	md = scinew MessageData( m, extraData, extraBytes );
      }
      else
	md = scinew MessageData( m );
      
      Log::log( DEBUG, string("[NetDispatchManager::fireCallback] Firing callback for message ") +
		MakeMessage::messageText( messageType ));
      
      

      // If the callback function isn't null, fire the callback
      if ( clients[q]->function != NULL ) {
         // DEBUGGING CODE -- DELETE
        cerr << "message type = " << MakeMessage::messageText( messageType ) << endl; 
        if( MakeMessage::messageText( messageType ) == "Multicast"){
          Multicast mc = *((Multicast *)(md->message));
          cerr << "In fireCallback - group = " << mc.getGroup() << ", port = " << mc.getPort() << endl; 
        }
        /*
        if( MakeMessage::messageText( messageType ) == "View Frame"){
          // print out rgb content of message data
          unsigned char * image = (unsigned char *)md->data;
          int numColoredPixels = 0;
          for ( int i = 0; i < 640 * 512 * 3; i+=3 ) {
            if((unsigned int)image[ i ] != 0 || (unsigned int)image[ i+1 ] != 0 || (unsigned int)image[ i+2 ] != 0){
              cerr << "<" << (unsigned int)image[ i ] << ", " << (unsigned int)image[ i+1 ] << ", " << (unsigned int)image[ i+2 ] << ">  ";
              numColoredPixels++;
            }
          }
          cerr << "**************************NUM COLORED PIXELS = " << numColoredPixels << endl;
	  
        }
        */
        // END DEBUGGING CODE
      
	clients[q]->function( clients[q]->object,
			      md );
	//delete m;
	md->message = NULL;
	delete md;
      }
      else 
	// Else, dump the message in the mailbox.
	result = clients[q]->mbox.trySend( *md );
      
      if ( !result )
	Log::log( DEBUG, "[NetDispatchManager::fireCallback] Dropped callback - mailbox full! ");

      Log::log( DEBUG, "[NetDispatchManager::fireCallback] Done firing callback" );
      
      /* If the callback isn't persistent, remove the callback from the 
	 list. */
      if ( clients[q]->persistent == false ) {
	deleteCallback( clients[q]->ID );
	Log::log( DEBUG, "[NetDispatchManager::fireCallback] Done deleting callback" );
      }
    }
    delete m;
  }
  
  /* If we cannot find the client, note that in log. */
  else {
    Log::log( MESSAGE, string("[NetDispatchManager::fireCallback] Message not requested by any module: ") +
	      MakeMessage::messageText( messageType ) );
  }
  delete goodData;
  cerr << "FireCallback: callback took " << p.time() * 1000.0 << " ms." << endl;
  p.stop();
  
  Log::log( LEAVE, "[NetDispatchManager::fireCallback] leaving, thread id = " + mkString((int) pthread_self()) );
}

char *
NetDispatchManager::findData( const char * data ) {
#if 0
  Log::log( ENTER, "[NetDispatchManager::findData] entered, thread id = " + mkString((int) pthread_self()) );
  /* Look for useful data in the given data by looking for the special
     end of data marker... */
  char * substring = strstr( data, NetInterface::dataMarker );
  
  /* If substring is null, then we didn't find the marker. */
  if ( substring == NULL )
    return substring;

  /* Now everything in data up to substring is (supposedly!) valid data.
     Make a copy, and return that copy. */
  char * returnString;
  int length = (int)(substring - data);
  
  returnString = scinew char[ length ];
  strncpy( returnString, data, length );
  return returnString;
#else
  int len = strlen( data );
  int markerLen = strlen( NetInterface::dataMarker );
  int i;
  bool found = false;
  for ( i = 0; i < len && !found; i++ ) {
    if ( data[i] == NetInterface::dataMarker[0] ) {
      found = true;
      for ( int j = 1; j < markerLen; j++ )
	if ( data[ i + j ] != NetInterface::dataMarker[j] ) {
	  found = false;
	  break;
	}
    }
  }
  if ( !found )
    return NULL;
  char * returnString = scinew char[ i+1 ];
  memcpy( returnString, data, i );
  returnString[ i ] = 0;
  return returnString;
  
  /*  if ( data )
    return strdup( data );
  else
    return NULL;
  */

  Log::log( LEAVE, "[NetDispatchManager::findData] leaving, thread id = " + mkString((int) pthread_self()) );
#endif
}

vector<clientRegister*>  
NetDispatchManager::searchCallbackList( const message_t type ) {
  Log::log( ENTER, "[NetDispatchManager::searchCallbackList] entered, thread id = " + mkString((int) pthread_self()) );
  vector<clientRegister*> returnval;

  // Iterate through the list looking for the particular message type
  list<clientRegister>::iterator i;

  for ( i = registeredClients.begin();
	i != registeredClients.end();
	i++ ) {

    if ( i->type == type )
      returnval.push_back( &(*i) );
  }

  Log::log( LEAVE, "[NetDispatchManager::searchCallbackList] leaving, thread id = " + mkString((int) pthread_self()) );
  return returnval;
}
  

}
