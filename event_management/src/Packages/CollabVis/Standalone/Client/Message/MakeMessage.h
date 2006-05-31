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
#include <Message/ViewFrame.h>
#include <Message/XDisplay.h>
#include <Message/Transfer.h>

namespace SemotusVisum {


/** Message Types */
typedef enum {
  HANDSHAKE,          
  MOUSE_MOVE,         
  SET_VIEWING_METHOD, 
  GET_CLIENT_LIST,    
  GET_Z_TEX,
  MULTICAST,
  GOODBYE,
  COMPRESSION,
  COLLABORATE,
  TRANSFER,
  CHAT,
  XDISPLAY,
  GROUP_VIEWER,
  VIEW_FRAME,
  UNKNOWN_M,
  ERROR_M
} message_t;

/**
 * This class parses the given data, creates and fills in the
 * appropriate message, and returns it. 
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class MakeMessage {
public:

  /**
   * Parses the given data, creates and fills in the appropriate
   * message, and returns it. The type parameter is set to the type of
   * the message, or UNKNOWN/ERROR on an error.
   *
   * @param data     Raw XML data
   * @param type     Enumerated message type
   * @return         New message, or NULL on error.
   */
  static MessageBase * makeMessage( void * data, message_t &type);

  /**
   *  Returns a textual description of the given message type.
   *
   * @param type  Enumerated message type
   * @return      Textual description of the message.
   */
  static inline const string messageText( message_t type );
  
private:
  
  /**
   *  Constructor - can't create them!
   *
   */
  MakeMessage() {}

  /**
   * Destructor - can't destroy them! 
   *
   */
  ~MakeMessage() {}
};


const string
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
  case CHAT:                return "Chat";
  case COLLABORATE:         return "Collaborate";
  case TRANSFER:            return "Transfer";
  case XDISPLAY:            return "XDisplay";
  case GROUP_VIEWER:        return "Group Viewer";
  case VIEW_FRAME:          return "View Frame";
  case UNKNOWN_M:           return "Unknown";
  case ERROR_M:             return "Error";
  default:                  return "This is not a message type";
  }
  
}
 

}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:27  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:11  simpson
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

