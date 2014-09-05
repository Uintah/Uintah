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

namespace SemotusVisum {


/**
 * This class provides the infrastructure to create, read, and serialize
 * a Multicast message.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class Multicast : public MessageBase {
public:

  
  /**
   *  Constructor.
   *
   */
  Multicast();

  /**
   *  Destructor.
   *
   */
  ~Multicast();

  /**
   *  Finishes serializing the message.
   *
   */
  void finish();

  /**
   *  Returns the multicast group.
   *
   */
  inline const string& getGroup() const { return group; }
  
  /**
   *  Sets the multicast group.
   *
   * @param group 
   */
  inline void   setGroup( const string group ) { this->group = group; }
  
  /**
   *  Returns the port
   *
   * @return Port number
   */
  inline int    getPort() const { return port; }
  
  /**
   * Returns the TTL setting
   *
   * @return TTL setting
   */
  inline int    getTTL() const { return ttl; }
  
  /**
   * Sets the disconnect parameter.
   *
   * @param disconnect    True to disconnect from multicast group.
   */
  inline void   setDisconnect( const bool disconnect ) {
    this->disconnect = disconnect;
    isDisconn = true;
  }
  
  /**
   * Returns true if the message is to disconnect.
   *
   * @return True if the message says to disconnect.
   */
  inline bool   getDisconnect() const { return disconnect; }

  
  /**
   *  Returns true if the message contains disconnect info.
   *
   * @return True if the message contains disconnect info.
   */
  inline bool   isDisconnect() const { return isDisconn; }
  
  /**
   *  Sets the confirm (will or won't join group) parameter.
   *
   * @param confirm       True to join, false to refuse.
   */
  inline void   setConfirm( const bool confirm ) { this->confirm = confirm; }
  
  /**
   *  Returns a Multicast message from the given raw data. 
   *
   * @param data   Raw input data.
   * @return       Multicast message, or NULL on error
   */
  static Multicast * mkMulticast( void * data );
  
protected:
  
  /** Multicast group */
  string group;

  /** Port */
  int    port;

  /** Time-to-live setting */
  int    ttl;

  /** True to disconnect */
  bool   disconnect;

  /** Are we refusing or agreeing to join the group? */
  bool   confirm;

  /** True if this message has disconnect info */
  bool   isDisconn;
};


}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:28  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:12  simpson
// Adding CollabVis files/dirs
//
// Revision 1.2  2001/08/08 01:58:05  luke
// Multicast working preliminary on Linux
//
// Revision 1.1  2001/06/05 20:53:48  luke
// Added driver and message for multicast
//
