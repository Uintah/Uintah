/*
 *
 * GroupViewerDriver: Tests GroupViewer message
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#include <Message/GroupViewer.h>

using namespace SemotusVisum;
  
/** Test XML String */
static const string XML1 = "<GroupViewer><add name=\"imageDefault\" viewer=\"Window0\"></add></GroupViewer>";

  /** Test XML String */
static const string XML2 = "<GroupViewer><subtract name=\"geomDefault\" viewer=\"Window1\"></subtract></GroupViewer>";

/** Test XML String */
static const string XML4 = "<GroupViewer><group name=\"Bar1\" viewer=\"Wind1\"></group><group name=\"wassabe\" viewer=\"wassapVindow1\"></group></GroupViewer>";

/**
 * Main function for driver.
 *
 * @param   argv      Arguments if needed
 *
 * @throws Exception  If there is a problem creating the message
 */	
int
main() {
  
  GroupViewer *g = new GroupViewer();
  
  g->finish();
  
  cerr << g->getOutput() << endl;
  
  // Test 1
  g = GroupViewer::mkGroupViewer( (void *)XML1.data() );
  
  if ( g == NULL )
    cerr << "Error creating GroupViewer in test 1" << endl;
  else 
    if ( !g->isListAdd() || g->isListSub() || g->isListFill() ||
	 strcasecmp(g->getGroupName(), "imageDefault") ||
	 strcasecmp(g->getGroupViewer(), "Window0") )
      cerr << "Error in GroupViewer in test 1" << endl;
  
  // Test 2
  g = GroupViewer::mkGroupViewer((void *) XML2.data() );
  
  if ( g == NULL )
    cerr << "Error creating GroupViewer in test 2" << endl;
  else 
    if ( g->isListAdd() || !g->isListSub() || g->isListFill() ||
	 strcasecmp(g->getGroupName(),"geomDefault") ||
	 strcasecmp(g->getGroupViewer(),"Window1") )
      cerr << "Error in GroupViewer in test 2" << endl;
  
  
  // Test 4
  g = GroupViewer::mkGroupViewer( (void *)XML4.data() );
  
  if ( g == NULL )
    cerr << "Error creating GroupViewer in test 4" << endl;
  else {
    if ( g->isListAdd() || g->isListSub() || !g->isListFill() )
      cerr << "Error in GetGroupViewer in test 4" << endl;
    vector<struct groupListItem> l = g->getGroupNames();
    if ( l.empty() ) 
      cerr << "Error in GroupViewer in test 4 - no list" << endl;
    string name = l[0].group;
    string viewer = l[0].viewer;

    if ( strcasecmp(name,"Bar1") || 
	 strcasecmp(viewer,"Wind1") )
      cerr << "Error in first group in test 4" << endl;
    name = l[1].group;
    viewer = l[1].viewer; 
    if ( strcasecmp(name,"wassabe") || 
	 strcasecmp(viewer,"wassapVindow1") )
      cerr << "Error in second group in test 4" << endl;
    
  }
  
}

