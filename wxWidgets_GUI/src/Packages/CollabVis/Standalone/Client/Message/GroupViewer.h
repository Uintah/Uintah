/*
 *
 * GroupViewer: Message that encapsulates a request for the currently
 *              available render groups and viewer windows.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: October 2001
 *
 */

#ifndef __GROUP_VIEWER_H_
#define __GROUP_VIEWER_H_

#include <Message/MessageBase.h>

#include <vector>

namespace SemotusVisum {

/**
 * Encapsulates the combination of a render group and server viewer.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
struct groupListItem {
  /** Render Group */
  string group;

  /** Server viewer */
  string viewer;

  /**
   * Constructor
   *
   * @param g   Render group  
   * @param v   Server viewer
   */
  groupListItem( const string &g, const string &v ) :
    group(g), viewer(v) {}
  
  /**
   *  Destructor
   *
   */
  ~groupListItem() {
  }
  
  /**
   * Returns true if the item is equal to this.
   *
   * @param g     Item to be tested
   * @return      True if parameter is textually equal.
   */
  inline bool equals( const groupListItem &g ) {
    return ( group == g.group ) && ( viewer == g.viewer );
  }
};


/**
 * This class provides the infrastructure to create, read, and serialize
 * a GroupViewer message.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class GroupViewer : public MessageBase {
public:

  /**
   *  Constructor.
   *
   */
  GroupViewer();

  /**
   *  Destructor. Deallocates all memory.
   *
   */
  ~GroupViewer();

  /**
   *  Finishes serializing the message.
   *
   */
  void finish();

  /**
   *  Returns true if the message adds a group/list pair.
   *
   * @return True if the message adds a group/list pair.
   */
  inline bool isListAdd() const { return listAdd; }
  
  /**
   *  Returns true if the message removes a group/list pair.
   *
   * @return True if the message removes a group/list pair.
   */
  inline bool isListSub() const { return listSub; }

  /**
   *  Returns true if the message contains multiple group/list pairs.
   *
   * @return True if the message contains multiple group/list pairs.
   */
  inline bool isListFill() const { return listFill; }
  
  /**
   * Returns the render group name
   *
   * @return Text of render group name
   */
  inline string getGroupName() { return groupName; }
  
  /**
   * Sets the render group name
   *
   * @param name  New group name.
   */
  inline void setGroupName( const string name ) { groupName = name; }
    
  /**
   * Returns the server viewer name
   *
   * @return Text of server viewer name
   */
  inline string getGroupViewer() { return groupViewer; }
  
  /**
   * Sets the server viewer name
   *
   * @param name  New server viewer name.
   */
  inline void setGroupViewer( string name ) { groupViewer = name; }
  
  /**
   * Returns the list of group name pairs, if multiple are sent.
   *
   * @return List of group/name pairs.
   */
  inline vector<groupListItem>& getGroupNames() { return groups; }
    
  /**
   * Returns a GroupViewer message from the given raw data.
   *
   * @param data  Raw data
   * @return      New message, or NULL on error.
   */
  static GroupViewer * mkGroupViewer( void * data );
  
protected:

  /** Are we adding to the list? */
  bool listAdd;
  /** Are we removing from the list? */
  bool listSub;
  
  /** Are we recreating the full list? */
  bool listFill;

  /** Group name */
  string groupName;

  /** Server viewer name */
  string groupViewer;
  
  /** List of group groups and viewers. */
  vector<struct groupListItem>   groups;
};

}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:26  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:11  simpson
// Adding CollabVis files/dirs
//
