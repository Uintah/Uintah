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

#include <vector>

namespace SemotusVisum {


/**
 * This class provides the infrastructure to create, read, and serialize
 * a GetClientList message.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class GetClientList : public MessageBase {
public:

  /**
   * Constructor. By default all messages are outgoing.
   *
   * @param request       
   */
  GetClientList( bool request = false );
  
  /**
   *  Destructor. Deallocates all memory.
   *
   */
  ~GetClientList();

  /**
   *  Finishes serializing the message.
   *
   */
  void finish();

  /**
   * Returns true if this message adds a client.
   *
   * @return True if the message adds a client.
   */
  inline bool isListAdd() const { return listAdd; }
  
  /**
   *  Returns true if this message removes a client.
   *
   * @return True if this message removes a client.
   */
  inline bool isListSub() const { return listSub; }
  
  /**
   * Returns true if this message contains the global client list.
   *
   * @return True if this message contains the global client list.
   */
  inline bool isListFill() const { return listFill; }

  
  /**
   * Returns true if this message modifies a client.
   *
   * @return True if this message modifies a client.
   */
  inline bool isListModify() const { return listModify; }
  
  /**
   * Returns the client name if relevant (not a full list).
   *
   * @return Client name.
   */
  inline string& getClientName() { return clientName; }

  
  /**
   * Sets the client name
   *
   * @param name   New client name
   */
  inline void setClientName( const string& name ) {
    clientName = name;
  }
  
  /**
   *  Returns the client IP address if relevant (not a full list).
   *
   * @return Client address.
   */
  inline string& getClientAddr() { return clientAddr; }
  
  /**
   * Sets the client address
   *
   * @param addr  New client address
   */
  inline void setClientAddr( const string& addr ) {
    clientAddr = addr;
  }
  
  /**
   *  Returns the client group if relevant (not a full list).
   *
   * @return Client group.
   *
   */
  inline string& getClientGroup() { return clientGroup; }

  
  /**
   * Sets the client rendering group
   *
   * @param group  New rendering group
   */
  inline void setClientGroup( const string& group ) {
    clientGroup = group;
  }
  
  /**
   * Gets the list of client names, addresses, and groups. The format is
   * Name1, Address1, Group1, Name2, Address2, Group2, ....
   *
   * @return List of client names, addresses, and groups.
   */
  inline vector<string>& getClientNames() { return clients; }
    
  /**
   * Returns a GetClientList message from the given raw data.
   *
   * @param data  Raw data
   * @return      New message, or NULL on error.
   */
  static GetClientList * mkGetClientList( void * data );
  
protected:

  /** What type of message is this? */
  bool listAdd, listSub, listModify, listFill;

  /** Client name */
  string clientName;

  /** Client address */
  string clientAddr;

  /** Client group */
  string clientGroup;
  
  /** List of client names and addresses. */
  vector<string>   clients;
  
};

}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:25  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:10  simpson
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
