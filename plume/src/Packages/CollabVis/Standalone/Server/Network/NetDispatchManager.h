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

#ifndef __NetDispatchManager_h_
#define __NetDispatchManager_h_

#include <list.h>

#include <Message/MessageBase.h>
#include <Message/MakeMessage.h>

#include <Network/NetInterface.h>
#include <Logging/Log.h>
#include <Thread/Mailbox.h>
#include <Malloc/Allocator.h>

#ifdef DEBUG
#undef DEBUG
#endif

namespace SemotusVisum {
namespace Network {


using namespace SCIRun;
using namespace Message;

//////////
// Callback function type.
typedef void (*callbackFunc)( void * obj, MessageData * input );

/**************************************
 
CLASS
   clientRegister
   
KEYWORDS
   Network
   
DESCRIPTION

   ClientRegister represents callback data. It contains a message tag,
   a mailbox, a notion of persistence, and a unique ID.
   
****************************************/

struct clientRegister {

  //////////
  // Constructor - Fills in the local data structors with the info
  // given. Mailbox constructor.  
  clientRegister( const message_t type,
		  Mailbox<MessageData>& mbox,
		  bool forever,
		  int ID) :
    type(type), mbox(mbox), function( NULL ), object( NULL ),
    persistent(forever), ID(ID) { }
  
  //////////
  // Constructor - Fills in the local data structors with the info
  // given. Function constructor.
  clientRegister( const message_t type,
		  callbackFunc function,
		  void * object,
		  bool forever,
		  int ID ) :
    type(type), mbox( clientRegister::nullMbox ),
    function( function ), object( object), persistent(forever), ID(ID) { }
  
  message_t              type;          // Message type.
  Mailbox<MessageData>&  mbox;          // Mailbox.
  callbackFunc           function;      // Callback function
  void *                 object;        // Object to call the function on.
  bool                   persistent;    // Is this a persistent callback?
  int                    ID;            // Unique ID for this callback.

private:
  static Mailbox<MessageData> nullMbox;
};


/**************************************
 
CLASS
   NetDispatchManager
   
KEYWORDS
   Network
   
DESCRIPTION

   NetDispatchManager handles incoming data from the network, and
   sends messages based on the tag in the beginning of the data.
   
****************************************/

class NetDispatchManager {
public:
  
  //////////
  // Returns the singleton instance of the network dispatch manager.
  inline static NetDispatchManager& getInstance() {
    if ( manager == NULL )
      manager = scinew NetDispatchManager();
    return *manager;
  }

  //////////
  // Registers a callback. Requires a tag, a message mailbox,
  // and a need for persistence (ie, is this a one-time callback?).
  // Returns an ID for the callback.
  int                        registerCallback( const message_t type,
					       Mailbox<MessageData>& mbox,
					       bool persistent=false );
  
  //////////
  // Registers a callback. Requires a tag, a callback function, an object
  // on which to call the function (this parameter may be NULL), 
  // and a need for persistence (ie, is this a one-time callback?).
  // Returns an ID for the callback.
  int                        registerCallback( const message_t type,
					       callbackFunc function,
					       void * object,
					       bool persistent=false );
  
  //////////
  // Deletes the callback associated with the given ID.
  void                       deleteCallback( const int ID );

  //////////
  // Deletes ALL CALLBACKS requesting the given type.
  void                       deleteCallback( const message_t type );

  //////////
  // (Potentially) Fires a callback. Data is the input data from the
  // network; client is the name of the client that delivered the data.
  void                       fireCallback( const char * data,
					   const int    numBytes,
					   const char * client );
  
protected:
  static NetDispatchManager *manager;  // The singleton instance
  static int                baseID;   // A base ID for callback registration.
  
  list<clientRegister>     registeredClients; // A list of registered clients.

  // Finds the useful data in the beginning of the data.
  char *           findData( const char * data );

  // Returns the client register structure for the client that is
  // registered to receive the given message type.
  clientRegister * searchCallbackList( const message_t type );

  // Constructor - can't make one!
  NetDispatchManager() { }

  // Destructor - clears the registered client list.
  ~NetDispatchManager() {
    registeredClients.clear();
  }
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:25  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:45  simpson
// Adding CollabVis files/dirs
//
// Revision 1.10  2001/10/11 16:38:08  luke
// Foo
//
// Revision 1.9  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.8  2001/08/01 00:16:06  luke
// Compiles on SGI. Fixed list allocation bug in NetDispatchManager
//
// Revision 1.7  2001/06/06 21:16:44  luke
// Callback functions now an option in NetDispatchManager
//
// Revision 1.6  2001/05/14 22:39:01  luke
// Fixed compiler warnings
//
// Revision 1.5  2001/05/14 18:07:56  luke
// Finished documentation
//
// Revision 1.4  2001/05/12 03:29:11  luke
// Now uses messages instead of XML. Also moved drivers to new location
//
// Revision 1.3  2001/04/05 22:28:00  luke
// Documentation done
//
// Revision 1.2  2001/04/04 21:45:29  luke
// Added NetDispatch Driver. Fixed bugs in NDM.
//
// Revision 1.1  2001/02/08 23:53:29  luke
// Added network stuff, incorporated SemotusVisum namespace
//
