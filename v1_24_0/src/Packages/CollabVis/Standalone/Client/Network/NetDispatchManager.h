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
#include <Network/dataItem.h>

#include <Thread/Mailbox.h>
#include <Malloc/Allocator.h>

#ifdef DEBUG
#undef DEBUG
#endif

namespace SemotusVisum {

using namespace SCIRun;

/** Callback function type. */
typedef void (*callbackFunc)( void * obj, MessageData * input );


/**
 * ClientRegister represents callback data. It contains a message tag,
 * a mailbox or callback function with associated object,
 * a notion of persistence, and a unique ID.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
struct clientRegister {

  /**
   *  Constructor - Fills in the local data structors with the info
   * given. Mailbox constructor.
   *
   * @param type       Message type.
   * @param mbox       Mailbox for callbacks
   * @param forever    Is this callback persistent?
   * @param ID         ID for callback.
   */
  clientRegister( const message_t type,
		  Mailbox<MessageData>& mbox,
		  bool forever,
		  int ID) :
    type(type), mbox(mbox), function( NULL ), object( NULL ),
    persistent(forever), ID(ID) { }
  

  /**
   * Constructor - Fills in the local data structors with the info
   * given. Function constructor.
   *
   * @param type       Message type.  
   * @param function   Callback function   
   * @param object     Object to pass to the callback function      
   * @param forever    Is this callback persistent?      
   * @param ID         ID for callback.    
   */
  clientRegister( const message_t type,
		  callbackFunc function,
		  void * object,
		  bool forever,
		  int ID ) :
    type(type), mbox( clientRegister::nullMbox ),
    function( function ), object( object), persistent(forever), ID(ID) { }

  /// Message type.
  message_t              type;

  /// Mailbox.
  Mailbox<MessageData>&  mbox;

  /// Callback function
  callbackFunc           function;

  /// Object to call the function on.
  void *                 object;

  /// Is this a persistent callback?
  bool                   persistent;

  /// Unique ID for this callback.
  int                    ID;            

private:
  /** Null mailbox for function callback structures. */
  static Mailbox<MessageData> nullMbox;
};


/**
 * NetDispatchManager handles incoming data from the network, and
 * sends messages based on the tag in the beginning of the data.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class NetDispatchManager {
public:
  
  /**
   * Returns the singleton instance of the network dispatch manager.
   *
   */
  inline static NetDispatchManager& getInstance() {
    if ( manager == NULL )
      manager = scinew NetDispatchManager();
    return *manager;
  }

  /**
   *  Registers a callback. Requires a tag, a message mailbox,
   * and a need for persistence (ie, is this a one-time callback?).
   *
   * @param type          Message type
   * @param mbox          Mailbox for callback data
   * @param persistent    Persistent callback?
   * @return Returns an ID for the callback.
   */
  int                        registerCallback( const message_t type,
					       Mailbox<MessageData>& mbox,
					       bool persistent=false );
  

  /**
   * Registers a callback. Requires a tag, a callback function, an object
   * on which to call the function (this parameter may be NULL), 
   * and a need for persistence (ie, is this a one-time callback?).
   *
   * @param type          Message type  
   * @param function      Callback function
   * @param object        Object to pass to the callback function
   * @param persistent     Persistent callback?   
   * @return Returns an ID for the callback.
   */
  int                        registerCallback( const message_t type,
					       callbackFunc function,
					       void * object,
					       bool persistent=false );
  
  /**
   * Deletes the callback associated with the given ID.
   *
   * @param ID    
   */
  void                       deleteCallback( const int ID );

  /**
   *  Deletes ALL CALLBACKS requesting the given type.
   *
   * @param type   Message type
   */
  void                       deleteCallback( const message_t type );

  /**
   * Fires a callback. Data is the input data from the
   * network; server is the name of the location (server/multicast) that
   * delivered the data.
   *
   * @param data      Raw data.
   * @param numBytes  Number of bytes of data
   * @param server    Origination of data (server/multicast)
   */
  void                       fireCallback( const char * data,
					   const int    numBytes,
					   const string server );
  
protected:
  /// The singleton instance
  static NetDispatchManager *manager;

  /// A base ID for callback registration.
  static int                baseID;   

  /// A list of registered clients.
  list<clientRegister>     registeredClients; 

  /// Finds the useful data in the beginning of the data.
  char *           findData( const char * data );


  /** Returns the client register structures for the client(s) that is(are)
      registered to receive the given message type. */
  vector<clientRegister*> searchCallbackList( const message_t type );
  
  /** Constructor - can't make one! */
  NetDispatchManager() { }

  /** Destructor - clears the registered client list. */
  ~NetDispatchManager() {
    registeredClients.clear();
  }
};

}

#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:32  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:01:00  simpson
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
