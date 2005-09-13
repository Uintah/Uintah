/*
 *
 * XDisplayDriver: Tests XDisplay message
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#include <Message/XDisplay.h>

using namespace SemotusVisum;

/** Test XML String */
static const string XML1 = "<getXDisplay response=\"Okay\"/>";

/** Test XML String */
static const string XML2 = "<getXDisplay response=\"Error\">Module locked!</getXDisplay>";

/** Test XML String */
static const string XML3 = "<moduleSetup><module name=\"Foo_3\" x=\"6\" y=\"66\" type=\"add\"><connection name=\"Mufasa_1\"/><connection name=\"Mufasa_4\"/></module></moduleSetup>";

/** Test XML String */
static const string XML4 = "<moduleSetup><module name=\"Foo_3\" x=\"6\" y=\"66\" type=\"modify\"><connection name=\"Mufasa_14\"/></module></moduleSetup>";

/** Test XML String */
static const string XML5 = "<moduleSetup><module name=\"Foo_3\" x=\"6\" y=\"66\" type=\"remove\"></module></moduleSetup>";

  /**
   * Main function for driver.
   *
   * @param   argv      Arguments if needed
   *
   * @throws Exception  If there is a problem creating the message
   */	
int
main() {

  XDisplay *x = new XDisplay();
  
  x->setDisplay("localhost:0.0");
  x->setModuleName("Foo_3");
  x->finish();
  
  cout << x->getOutput() << endl;
  
  
  // Test 1
  x = XDisplay::mkXDisplay( (void *)XML1.data() );
  
  if ( x == NULL )
    cerr << "Error creating XDisplay in test 1" << endl;
  else 
    if ( x->isModuleSetup() || !x->isDisplayResponse() || !x->isResponseOkay() )
      cerr << "Error in XDisplay in test 1" << endl;
  
  // Test 2
  x = XDisplay::mkXDisplay( (void *)XML2.data() );
  
  if ( x == NULL )
    cerr << "Error creating XDisplay in test 2" << endl;
  else 
    if ( x->isModuleSetup() || !x->isDisplayResponse() || 
	 x->isResponseOkay() || 
	 strcasecmp( x->getErrorText(), "Module locked!") )
      cerr << "Error in XDisplay in test 2" << endl;

  
  // Test 3
    x = XDisplay::mkXDisplay( (void *)XML3.data() );
    
    if ( x == NULL )
      cerr << "Error creating XDisplay in test 3" << endl;
    else {
      if ( !x->isModuleSetup() || x->isDisplayResponse() || 
	   x->getNumModules() != 1 ) {
	cerr << "Error 1 in XDisplay in test 3" << endl;
	cerr << x->isModuleSetup() << " " << x->isDisplayResponse() <<
	  " " << x->getNumModules()  << endl;
      }

      if ( x->getModules().empty() )
	cerr << "Error in XDisplay in test 3 - no module" << endl;
      else {
	Module m = x->getModules()[0];
      
	if ( m.isRemoved() || m.isModification() ) 
	  cerr << "Error in XDisplay in test 3 - bad mod params" << endl;

	vector<string> l = m.getConnections();
	if ( l.empty() || l.size() < 2 ) 
	  cerr << "Error in XDisplay in test 3 - no conn list" << endl;
	else {
	  string conn1 = l[0];
	  string conn2 = l[1];
	  
	  if ( strcasecmp(conn1,"Mufasa_1") || 
	       strcasecmp(conn2,"Mufasa_4") )
	    cerr << "Error in module connections in test 3" << endl;
	}
      }
    }

    // Test 4
    x = XDisplay::mkXDisplay( (void *)XML4.data() );
    
    if ( x == NULL )
      cerr << "Error creating XDisplay in test 4" << endl;
    else {
      if ( !x->isModuleSetup() || x->isDisplayResponse() || 
	   x->getNumModules() != 1 ) {
	cerr << "Error 1 in XDisplay in test 4" << endl;
	cerr << x->isModuleSetup() << " " << x->isDisplayResponse() <<
			    " " << x->getNumModules() << endl;
      }
      
      if ( x->getModules().empty() )
	cerr << "Error in XDisplay in test 6 - no module" << endl;
      else {
	Module m = x->getModules()[0];
	
	
	if ( m.isRemoved() || !m.isModification() ) 
	  cerr << "Error in XDisplay in test 4 - bad mod params" << endl;
	
	vector<string> l = m.getConnections();
	if ( l.empty() ) 
	  cerr << "Error in XDisplay in test 4 - no conn list" << endl;
	else {
	  
	  string conn1 = l[0];
	  
	  if ( strcasecmp(conn1,"Mufasa_14") )
	    cerr << "Error in module connection in test 4" << endl;
	}
      }
    }
    
    
    // Test 5
    x = XDisplay::mkXDisplay( (void *)XML5.data() );
    
    if ( x == NULL )
      cerr << "Error creating XDisplay in test 5" << endl;
    else {
      if ( !x->isModuleSetup() || x->isDisplayResponse() || 
	   x->getNumModules() != 1 ) {
	cerr << "Error 1 in XDisplay in test 5" << endl;
	cerr << x->isModuleSetup() << " " << x->isDisplayResponse() <<
	  " " << x->getNumModules() << endl;
      }
      
      if ( x->getModules().empty() )
	cerr << "Error in XDisplay in test 5 - no module" << endl;
      else {
	Module m = x->getModules()[0];
	
	if ( !m.isRemoved() || m.isModification() ) 
	  cerr << "Error in XDisplay in test 5 - bad mod params" << endl;
	
      }
    
    }

}
