/*
 *
 * MakeMessage: Message that encapsulates a change in the viewing
 *                   method (or viewing parameters).
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: May 2001
 *
 */

#ifndef __MAKE_MESSAGE_H_
#define __MAKE_MESSAGE_H_

#include <Message/Chat.h>
#include <Message/Collaborate.h>
#include <Message/Compression.h>
#include <Message/MessageBase.h>
#include <Message/GetClientList.h>
#include <Message/GetZTex.h>
#include <Message/Goodbye.h>
#include <Message/GroupViewer.h>
#include <Message/Handshake.h>
#include <Message/MouseMove.h>
#include <Message/Multicast.h>
#include <Message/SetViewingMethod.h>
#include <Message/XDisplay.h>
#include <Message/Transfer.h>

namespace SemotusVisum {
namespace Message {

//////////
// Message Types
typedef enum {
  HANDSHAKE,
  MOUSE_MOVE,
  SET_VIEWING_METHOD,
  GET_CLIENT_LIST,
  GET_Z_TEX,
  MULTICAST,
  GOODBYE,
  COMPRESSION,
  TRANSFER,
  COLLABORATE,
  CHAT,
  XDISPLAY,
  GROUP_VIEWER,
  UNKNOWN,
  ERROR
} message_t;

/**************************************
 
CLASS
   MakeMessage
   
KEYWORDS
   Message
   
DESCRIPTION

   This class parses the given data, creates and fills in the
   appropriate message, and returns it. 
      
****************************************/
class MakeMessage {
public:

  //////////
  // Parses the given data, creates and fills in the appropriate
  // message, and returns it. The type parameter is set to the type of
  // the message, or UNKNOWN/ERROR on an error. 
  static inline MessageBase * makeMessage( void * data, message_t &type);

  ///////////
  // Returns a textual description of the given message type.
  static inline const char * messageText( message_t type );
  
private:
  
  //////////
  // Constructor - can't create them!
  MakeMessage() {}

  //////////
  // Destructor - can't destroy them!
  ~MakeMessage() {}
};

MessageBase *
MakeMessage::makeMessage( void * data, message_t &type ) {

  if ( !data ) {
    type = ERROR;
    return NULL;
  }

  /* Parse the XML */
  char * input = strdup( (char *)data );
  //  cerr << "Did strdup" << endl;
  XMLI::initialize();
  
  XMLReader reader( input );
  reader.parseInputData();
  	    
  char * tag = XMLI::getChar( reader.nextElement() );
  //cerr << "Parsed XML" << endl;
  delete input;
  
  if ( tag == NULL ) {
    type = ERROR;
    return NULL;
  }
  
  /* Switch among the different types of messages */
  if ( !strcasecmp( tag, "GetClientList" ) ) {
    type = GET_CLIENT_LIST;
    return GetClientList::mkGetClientList( data );
  }
  else if ( !strcasecmp( tag, "GetZTex" ) ) {
    type = GET_Z_TEX;
    return GetZTex::mkGetZTex( data );
  }
  else if ( !strcasecmp( tag, "Handshake" ) ) {
    type = HANDSHAKE;
    return Handshake::mkHandshake( data );
  }
  else if ( !strcasecmp( tag, "MouseMove" ) ) {
    type = MOUSE_MOVE;
    return MouseMove::mkMouseMove( data );
  }
  else if ( !strcasecmp( tag, "SetViewingMethod" ) ) {
    type = SET_VIEWING_METHOD;
    return SetViewingMethod::mkSetViewingMethod( data );
  }
  else if ( !strcasecmp( tag, "Multicast" ) ) {
    type = MULTICAST;
    return Multicast::mkMulticast( data );
  }
  else if ( !strcasecmp( tag, "Goodbye" ) ) {
    type = GOODBYE;
    return Goodbye::mkGoodbye( data );
  }
  else if ( !strcasecmp( tag, "Compression" ) ) {
    type = COMPRESSION;
    return Compression::mkCompression( data );
  }
  else if ( !strcasecmp( tag, "Transfer" ) ) {
    type = TRANSFER;
    return Transfer::mkTransfer( data );
  }
  else if ( !strcasecmp( tag, "Chat" ) ) {
    type = CHAT;
    return Chat::mkChat( data );
  }
  else if ( !strcasecmp( tag, "Collaborate" ) ) {
    type = COLLABORATE;
    return Collaborate::mkCollaborate( data );
  }
  else if ( !strcasecmp( tag, "getXDisplay" ) ) {
    type = XDISPLAY;
    return XDisplay::mkXDisplay( data );
  }
  else if ( !strcasecmp( tag, "GroupViewer" ) ) {
    type = GROUP_VIEWER;
    return GroupViewer::mkGroupViewer( data );
  }
  else {
    //Log::log( Logging::DEBUG, "Unknown tag: " + mkString(tag) );
    type = UNKNOWN;
    return NULL;
  }
}

const char *
MakeMessage::messageText( message_t type ) { 
  switch (type) {
  case HANDSHAKE:           return "Handshake";
  case MOUSE_MOVE:          return "Mouse Move";
  case SET_VIEWING_METHOD:  return "Set Viewing Method";
  case GET_CLIENT_LIST:     return "Get Client List";
  case GET_Z_TEX:           return "Get ZTex";
  case MULTICAST:           return "Multicast";
  case GOODBYE:             return "Goodbye";
  case COMPRESSION:         return "Compression";
  case TRANSFER:            return "Transfer";
  case CHAT:                return "Chat";
  case COLLABORATE:         return "Collaborate";
  case XDISPLAY:            return "XDisplay";
  case GROUP_VIEWER:        return "Group Viewer";
  case UNKNOWN:             return "Unknown";
  case ERROR:               return "Error";
  default:                  return "This is not a message type";
  }
  
}
 
}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:19  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:04  simpson
// Adding CollabVis files/dirs
//
// Revision 1.6  2001/10/03 17:59:19  luke
// Added XDisplay protocol
//
// Revision 1.5  2001/09/23 02:24:11  luke
// Added collaborate message
//
// Revision 1.4  2001/07/31 22:48:32  luke
// Pre-SGI port
//
// Revision 1.3  2001/07/16 20:29:29  luke
// Updated messages...
//
// Revision 1.2  2001/06/05 20:53:48  luke
// Added driver and message for multicast
//
// Revision 1.1  2001/05/28 18:04:25  luke
// Added MakeMessage functionality
//
