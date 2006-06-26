#include <Message/GroupViewer.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>
#include <Logging/Log.h>

namespace SemotusVisum {

GroupViewer::GroupViewer() : listAdd(false), listSub(false),
			     listFill(false) {

}


GroupViewer::~GroupViewer() {
  groups.clear();
}

void
GroupViewer::finish() {

  if ( finished )
    return;
  
  /* Create an XML GroupViewer document */
  XMLWriter writer;

  // Start a new document
  writer.newDocument();

  // Create a 'groupviewer' document
  Attributes attributes;
  
  writer.addElement("GroupViewer" );
  writer.addElement("request", attributes, "New List" );

  mkOutput( writer.writeOutputData() );
  finished = true;
  
}

GroupViewer *
GroupViewer::mkGroupViewer( void * data ) {
  
  GroupViewer * g;
   // Make an XML Reader
  XMLReader reader( (char *) data );

  reader.parseInputData();

  // Start building a set viewing message
  String element;

  element = reader.nextElement(); // GroupViewer
  if ( element == NULL )
    return NULL;

  element = reader.nextElement(); // type
  if ( element == NULL )
    return NULL;

  Attributes attributes = reader.getAttributes();
  string name = attributes.getAttribute("name");
  string viewer = attributes.getAttribute("viewer");
  string s = XMLI::getChar(element);
  if ( name.empty() ) {
    Log::log( ERROR,
	      "Missing name in groupviewer");
    return NULL;
  }
  if ( viewer.empty() ) {
    Log::log( ERROR,
	      "Missing viewer in groupviewer" );
    return NULL;
  }    
  g = scinew GroupViewer();
  
  /* If an add, add the group to the list. */
  if ( s == "Add" || s == "add" ) {
    g->listAdd = true;
    g->setGroupName( name );
    g->setGroupViewer( viewer );
  }
  /* If a subtract, remove the group from the list. */
  else if ( s == "Subtract" || s == "subtract" ) {
    g->listSub = true;
    g->setGroupName( name );
    g->setGroupViewer( viewer );
  }
  else if ( s == "group" ) {
      
    /* Clear the list. */
    g->groups.clear();
    g->listFill = true;
    
    /* Refill the list. */
    while ( element != NULL ) {
      
      g->groups.push_back( groupListItem(name,viewer) );
      
      element = reader.nextElement();
      
      attributes = reader.getAttributes();
      
      name   = attributes.getAttribute("name");
      viewer = attributes.getAttribute("viewer");
      
      if ( name.empty() ) {
	Log::log( ERROR,
		  "Missing name in groupviewer");
	return NULL;
      }
      if ( viewer.empty() ) {
	Log::log( ERROR,
		  "Missing viewer in groupviewer" );
	return NULL;
      }    
    }
  }
  else {
    Log::log( ERROR, "Unknown tag in clientList: " + s );
    return NULL;
  }
  
  return g;
}

}
//
// $Log$
// Revision 1.1  2003/07/22 20:59:26  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:10  simpson
// Adding CollabVis files/dirs
//
