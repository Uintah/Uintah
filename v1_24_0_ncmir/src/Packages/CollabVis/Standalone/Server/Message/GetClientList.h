/*
 *
 * GetClientList: Message that encapsulates a request for the currently
 *                connected clients.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: May 2001
 *
 */

#ifndef __GET_CLIENT_LIST_H_
#define __GET_CLIENT_LIST_H_

#include <Message/MessageBase.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>

#include <vector>

namespace SemotusVisum {
namespace Message {

struct clientListItem {
  char * name;
  char * address;
  char * group;
  clientListItem( char * n, char * a, char * g ) :
    name(n), address(a), group(g) {}
  ~clientListItem() {
    delete name;
    delete address;
    delete group;
  }
};

/**************************************
 
CLASS
   GetClientList
   
KEYWORDS
   GetClientList, Message
   
DESCRIPTION

   This class provides the infrastructure to create, read, and serialize
   a GetClientList message.
   
****************************************/
class GetClientList : public MessageBase {
public:

  //////////
  // Constructor. By default all messages are outgoing.
  GetClientList( bool request = false );

  //////////
  // Destructor. Deallocates all memory.
  ~GetClientList();

  //////////
  // Finishes serializing the message.
  void finish();

  //////////
  // Adds the given client name/address/group to the internal list.
  void addClient( const char * name, const char * address,
		  const char * group=NULL);

  //////////
  // Sets the given client as an 'add'.
  void clientAdded( const char * name, const char * address );

  //////////
  // Sets the given client as a 'subtract'.
  void clientSubtracted( const char * name, const char * address );

  /////////
  // Sets the given client as a 'modify'.
  void clientModified( const char * name, const char * address,
		       const char * group );
  //////////
  // Returns a list of client names, addresses, and groups
  inline vector<struct clientListItem*>& getClients() { return clientList; }
  
  //////////
  // Returns true if this is a request; else returns false.
  inline bool isRequest( ) const { return request; }
  
  //////////
  // Returns a GetClientList message from the given raw data.
  static GetClientList * mkGetClientList( void * data );
  
protected:

  // True if this message is a request.
  bool                  request;

  // List of client names and addresses.
  vector<struct clientListItem*>   clientList;
  
  // 0 if add, 1 if sub, 2 if modify, -1 if not set.
  int addSub;

  // Client name for add/subtract.
  char * name;

  // Client address for add/subtract.
  char * address;

  // Client group for modify
  char * group;
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:18  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:02  simpson
// Adding CollabVis files/dirs
//
// Revision 1.4  2001/07/16 20:29:29  luke
// Updated messages...
//
// Revision 1.3  2001/05/14 19:04:52  luke
// Documentation done
//
// Revision 1.2  2001/05/12 02:14:16  luke
// Switched Message base class to MessageBase
//
// Revision 1.1  2001/05/11 20:06:02  luke
// Initial coding of Message abstraction. Documentation not yet done.
//
