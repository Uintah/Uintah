/*
 *
 * Multicast: Message that encapsulates multicast requests, disconnects, etc.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#ifndef __MULTICAST_H_
#define __MULTICAST_H_

#include <Message/MessageBase.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>

namespace SemotusVisum {
namespace Message {

/**************************************
 
CLASS
   Multicast
   
KEYWORDS
   Multicast, Message
   
DESCRIPTION

   This class provides the infrastructure to create, read, and serialize
   a Multicast message.
   
****************************************/
class Multicast : public MessageBase {
public:

  //////////
  // Constructor. By default, all messages are outgoing.
  Multicast( bool request = true );

  //////////
  // Destructor.
  ~Multicast();

  //////////
  // Finishes serializing the message.
  void finish();

  //////////
  // Returns true if this is a request; else returns false.
  inline bool isRequest( ) const { return request; }

  //////////
  // Set the parameters for the multicast message.
  void   setParams( const char * group, int port, int ttl=-1 );

  //////////
  // Get the parameters from the multicast message.
  inline void getParams( char * &group, int &port, int &ttl ) {
    group = this->group; port = this->port; ttl = this->ttl;
  }

  //////////
  // Returns true if this is a 'disconnect' message.
  inline bool isDisconnect() const { return disconnect; }

  //////////
  // Sets the 'disconnect' parameter in the message.
  inline void setDisconnect( bool dis ) { disconnect = dis; }
  
  //////////
  // Returns true if this is an okay message and the response was yes;
  // else returns false.
  inline bool isOkay() const { return okay; }
  
  //////////
  // Returns a Multicast message from the given raw data.
  static Multicast * mkMulticast( void * data );
  
protected:
  // True if this message is a confirmation and the answer is yes.
  bool   okay;
  
  // True if this message is a disconnect message;
  bool   disconnect;
  
  // True if this message is a request.
  bool   request;

  // Multicast group
  char * group;

  // Multicast port
  int    port;

  // Multicast time to live
  int    ttl;
  
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:20  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:05  simpson
// Adding CollabVis files/dirs
//
// Revision 1.2  2001/08/08 01:58:05  luke
// Multicast working preliminary on Linux
//
// Revision 1.1  2001/06/05 20:53:48  luke
// Added driver and message for multicast
//
