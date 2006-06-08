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

namespace SemotusVisum {
namespace Network {

using namespace Logging;
using namespace Message;

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
  cerr << "In NetDispatchManager::registerCallback, thread id is " << pthread_self() << endl;

  char buffer[ 1000 ];
  snprintf( buffer, 1000, "Adding callback for message %s",
	    MakeMessage::messageText( tag ) );
  Log::log( Logging::DEBUG, buffer );
  clientRegister * cr = scinew clientRegister( tag,
					    mbox,
					    persistent,
					    baseID++ );

  // Add the data to the list of registered clients
  registeredClients.push_front( *cr );
  cerr << "End of NetDispatchManager::registerCallback, thread id is " << pthread_self() << endl;
  return baseID - 1;
}

int
NetDispatchManager::registerCallback( message_t tag,
				      callbackFunc function,
				      void * object,
				      bool persistent ) {
  cerr << "In NetDispatchManager::registerCallback with object, tag is " << tag << ", thread id is " << pthread_self() << endl;
  char buffer[ 256 ];
  snprintf( buffer, 256, "Adding callback for message %s",
	    MakeMessage::messageText( tag ) );
  Log::log( Logging::DEBUG, buffer );
  
  clientRegister * cr = scinew clientRegister( tag,
					    function,
					    object,
					    persistent,
					    baseID++ );

  // Add the data to the list of registered clients
  registeredClients.push_front( *cr );

  cerr << "End of NetDispatchManager::registerCallback with object, thread id is " << pthread_self() << endl;
  return baseID - 1;
}

void                       
NetDispatchManager::deleteCallback( const int ID ) {
  cerr << "In NetDispatchManager::deleteCallback, thread id is " << pthread_self() << endl;
  Log::log( Logging::DEBUG, "Deleting callback" );
  // Iterate through the list looking for the particular ID
  list<clientRegister>::iterator i;
  
  for ( i = registeredClients.begin();
	i != registeredClients.end();
	i++ )
    if ( i->ID == ID ) {
      registeredClients.erase( i );
      return;
    }
  cerr << "End of NetDispatchManager::deleteCallback, thread id is " << pthread_self() << endl;
}

void                       
NetDispatchManager::deleteCallback( const message_t type ) {
  cerr << "In NetDispatchManager::deleteCallback, thread id is " << pthread_self() << endl;
  Log::log( Logging::DEBUG, "Deleting callback" );
  // Iterate through the list looking for the particular type.
  list<clientRegister>::iterator i;
  
  for ( i = registeredClients.begin();
	i != registeredClients.end();
	i++ )
    //if ( !strcmp( i->type, type ) )
    if ( i->type == type )
      registeredClients.erase( i );
  cerr << "End of NetDispatchManager::deleteCallback, thread id is " << pthread_self() << endl;
}

void                       
NetDispatchManager::fireCallback( const char * data,
				  const int numBytes,
				  const char * client) {
  cerr << "In NetDispatchManager::fireCallback, thread id is " << pthread_self() << endl;

  char buffer[ 1000 ];
  snprintf( buffer, 256, "Got %d bytes of data to fire callback. Data:",
	    numBytes );
  Log::log( Logging::DEBUG, buffer );
  Log::log( Logging::DEBUG, data );
  
  /* Get valid data */
  char * goodData = findData( data );
  //cerr << "Found good data: " << (void *)goodData << endl;
  if ( goodData == NULL ) { 
    // No good data here
    Log::log( Logging::ERROR,
	      "Did not find any valid data from network!" );
    return;
  }

  message_t messageType;
  //cerr << "Building message" << endl;
  MessageBase *m = MakeMessage::makeMessage( (void *)goodData, messageType );
  //cerr << "Done" << endl;
  
  /* Check message */
  if ( m == NULL ) {
   if ( messageType == Message::ERROR ) {
     Log::log( Logging::ERROR, "Error in building message!" );
     Log::log( Logging::DEBUG, goodData );
   }
   else if ( messageType == Message::UNKNOWN) {
     snprintf( buffer, 1000, "Unknown message type from client %s",
	       client );
     Log::log( Logging::ERROR, buffer );
     Log::log( Logging::DEBUG, goodData );
   }
   else {
     snprintf( buffer, 1000, "BUG - '%s' message type, but no message!",
	       MakeMessage::messageText( messageType ) );
     Log::log( Logging::ERROR, buffer );
     Log::log( Logging::DEBUG, goodData );
   }
   return;
 }
  else {
    snprintf( buffer, 1000, "Got a '%s' message",
	      MakeMessage::messageText( messageType ) );
    Log::log( Logging::DEBUG, buffer );
  }
  
  /* Search the callback list for a client requesting that tag */
  clientRegister *clientCallback = searchCallbackList( messageType );
  
  /* If we find the client, add a message to the mailbox (or fire the
     callback function. */
  if ( clientCallback != NULL ) {
    bool result;
    MessageData *md = scinew MessageData( m, client );

    snprintf( buffer, 1000,
	      "Firing callback for message %s from client %s",
	      MakeMessage::messageText( messageType ), client );
    Log::log( Logging::DEBUG, buffer );
    
    // If the callback function isn't null, fire the callback
    if ( clientCallback->function != NULL ) {
     
      clientCallback->function( clientCallback->object,
				md );
      delete m;
      delete md;
    }
    else {
      // Else, dump the message in the mailbox.
      result = clientCallback->mbox.trySend( *md );

      if ( !result ){
        Log::log( Logging::DEBUG, "Dropped callback - mailbox full! ");
      }
    }

    Log::log( Logging::DEBUG, "Done firing callback" );
    /* If the callback isn't persistent, remove the callback from the 
       list. */
    if ( clientCallback->persistent == false ) {
      deleteCallback( clientCallback->ID );
      Log::log( Logging::DEBUG, "Done deleting callback" );
    }
  }

  /* If we cannot find the client, note that in log. */
  else {
    char buffer[ 1000 ];
    snprintf( buffer, 1000, "Message not requested by any module: %s",
	      MakeMessage::messageText( messageType ) );
    Log::log( Logging::MESSAGE, buffer );
  }
  cerr << "End of NetDispatchManager::fireCallback, thread id is " << pthread_self() << endl;
}

char *
NetDispatchManager::findData( const char * data ) {
  cerr << "In NetDispatchManager::findData, thread id is " << pthread_self() << endl;
#if 0
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
  cerr << "End of NetDispatchManager::findData, thread id is " << pthread_self() << endl;
#else
  if ( data )
    return strdup( data );
  else
    return NULL;
#endif
}

clientRegister *  
NetDispatchManager::searchCallbackList( const message_t type ) {
  cerr << "In NetDispatchManager::searchCallbackList, thread id is " << pthread_self() << endl;  
  // Iterate through the list looking for the particular message type
  list<clientRegister>::iterator i;

  for ( i = registeredClients.begin();
	i != registeredClients.end();
	i++ ) {

    //cerr << "Callback type is " << i->type << endl;
    if ( i->type == type ) {
      cerr << "End of NetDispatchManager::searchCallbackList, thread id is " << pthread_self() << endl;
      return &(*i);
    }
  }

  // None found. Return NULL.
  cerr << "End of NetDispatchManager::searchCallbackList returning NULL, thread id is " << pthread_self() << endl;  
  return NULL;
}

}
}
