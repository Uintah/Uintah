/*
 *
 * ChatDriver: Tests Chat message
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#include <Message/Chat.h>

using namespace SemotusVisum;
static string XML = "<chat client=\"foo@foo.com\">Sent chat</chat>";

int
main() {
  Chat c;

  c.setText( "Chat, blah, blah" );
  c.finish();
  cout << c.getText() << endl;

  Chat *cc;
  cc = Chat::mkChat( (char *)XML.data() );
  
  if ( cc == NULL )
    cerr << "Error in mkChat" << endl;
  else {
    cout << "Client: " << cc->getName() <<
      "\tChat text: " << cc->getText() << endl;
  }
  
}

//
// $Log$
// Revision 1.1  2003/07/22 20:59:13  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 20:50:44  simpson
// Adding CollabVis files/dirs
//
// Revision 1.1  2001/07/31 23:02:07  luke
// Added chat driver
//

