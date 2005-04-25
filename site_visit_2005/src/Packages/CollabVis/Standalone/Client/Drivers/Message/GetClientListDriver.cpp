/*
 *
 * GetClientListDriver: Tests GetClientList message
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#include <Message/GetClientList.h>

using namespace SemotusVisum;

/** Test XML String */
  static const string XML1 = "<clientList><add name=\"foo\" address=\"bar.com\"></add></clientList>";

  /** Test XML String */
  static const string XML2 = "<clientList><subtract name=\"foo\" address=\"bar.com\"></subtract></clientList>";

  /** Test XML String */
  static const string XML3 = "<clientList><modify name=\"foo\" address=\"bar.com\" group=\"greatRenderer\"></modify></clientList>";

  /** Test XML String */
  static const string XML4 = "<clientList><client name=\"foo\" address=\"bar.com\" group=\"HoochieMamaRendererGroup\"></client><client name=\"wassabe\" address=\"wassaaaap.com\"></client></clientList>";

/**
 * Main function for driver.
 *
 * @param   argv      Arguments if needed
 *
 * @throws Exception  If there is a problem creating the message
 */	
int
main() {
  
  GetClientList *g = scinew GetClientList();
  
  g->finish();

  cout << g->getOutput() << endl;
  delete g;
  
  // Test 1
  g = GetClientList::mkGetClientList( (void *)XML1.data() );
  if ( g == NULL )
    cerr << "Error creating GetClientList in test 1" << endl;
  else 
    if ( !g->isListAdd() || g->isListSub() || g->isListModify() || 
	 g->isListFill() ||
	 strcasecmp(g->getClientName(), "foo") ||
	 strcasecmp(g->getClientAddr(), "bar.com") ) {
      cerr << "Error in GetClientList in test 1" << endl;
    }
  
  // Test 2
  g = GetClientList::mkGetClientList( (void *)XML2.data() );
  
  if ( g == NULL )
    cerr << "Error creating GetClientList in test 2" << endl;
  else 
    if ( g->isListAdd() || !g->isListSub() || g->isListModify() || 
	 g->isListFill() ||
	 strcasecmp(g->getClientName(), "foo") ||
	 strcasecmp(g->getClientAddr(), "bar.com") )
      cerr <<  "Error in GetClientList in test 2" << endl;
  
  // Test 3
  g = GetClientList::mkGetClientList( (void *)XML3.data() );
  
  if ( g == NULL )
    cerr << "Error creating GetClientList in test 3" << endl;
  else 
    if ( g->isListAdd() || g->isListSub() || !g->isListModify() || 
	 g->isListFill() ||
	 strcasecmp(g->getClientName(), "foo") ||
	 strcasecmp(g->getClientAddr(), "bar.com") ||
	 strcasecmp(g->getClientGroup(), "greatRenderer") )
      cerr << "Error in GetClientList in test 3" << endl;
  
  // Test 4
  g = GetClientList::mkGetClientList( (void *)XML4.data() );

  if ( g == NULL )
    cerr <<  "Error creating GetClientList in test 4" << endl;
  else {
    if ( g->isListAdd() || g->isListSub() || !g->isListFill() )
      cerr << "Error in GetClientList in test 4" << endl;
    vector<string> &t = g->getClientNames();
    if ( t.empty() ) 
      cerr << "Error in GetClientList in test 4 - no list" << endl;
    string name = t[0];
    string addr = t[1];
    string group = t[2];

    if ( strcasecmp(name, "foo") || 
	 strcasecmp(addr, "bar.com") ||
	 strcasecmp(group, "HoochieMamaRendererGroup") )
      cerr << "Error in first client in test 4" << endl;
    name = t[3];
    addr = t[4];
    if ( strcasecmp(name, "wassabe") || 
	 strcasecmp(addr, "wassaaaap.com") )
      cerr << "Error in second client in test 4"  << endl;
    
  }
  
}
