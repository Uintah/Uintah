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
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>

#include <vector>

namespace SemotusVisum {
namespace Message {

struct groupListItem {
  char * group;
  char * viewer;
  groupListItem( char * g, char * v ) :
    group(g), viewer(v) {}
  ~groupListItem() {
    delete group;
    delete viewer;
  }
};

/**************************************
 
CLASS
   GroupViewer
   
KEYWORDS
   GroupViewer, Message
   
DESCRIPTION

   This class provides the infrastructure to create, read, and serialize
   a GroupViewer message.
   
****************************************/
class GroupViewer : public MessageBase {
public:

  //////////
  // Constructor. By default all messages are outgoing.
  GroupViewer( bool request = false );

  //////////
  // Destructor. Deallocates all memory.
  ~GroupViewer();

  //////////
  // Finishes serializing the message.
  void finish();

  //////////
  // Adds the given group/viewer to the internal list.
  void addGroup( const char * group, const char * viewer);

  //////////
  // Sets the given group as an 'add'.
  void groupAdded( const char * group, const char * viewer );

  //////////
  // Sets the given group as a 'subtract'.
  void groupSubtracted( const char * group, const char * viewer );

  //////////
  // Returns a list of groups and viewers
  inline vector<struct groupListItem*>& getGroups() { return groupList; }
  
  //////////
  // Returns true if this is a request; else returns false.
  inline bool isRequest( ) const { return request; }
  
  //////////
  // Returns a GroupViewer message from the given raw data.
  static GroupViewer * mkGroupViewer( void * data );
  
protected:

  // True if this message is a request.
  bool                  request;

  // List of group groups and vieweres.
  vector<struct groupListItem*>   groupList;
  
  // 0 if add, 1 if sub, -1 if not set.
  int addSub;

  // Group for add/subtract.
  char * group;

  // Group viewer for add/subtract.
  char * viewer;
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:19  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:03  simpson
// Adding CollabVis files/dirs
//
