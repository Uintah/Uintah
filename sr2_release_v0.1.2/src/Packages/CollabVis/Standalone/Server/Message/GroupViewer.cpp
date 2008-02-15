#include <Message/GroupViewer.h>

namespace SemotusVisum {
namespace Message {

using namespace XML;

GroupViewer::GroupViewer( bool request ) : request(request),
					   addSub( -1 ),
					   group( NULL ),
					   viewer( NULL ) {

}


GroupViewer::~GroupViewer() {
  if ( group ) delete group;
  if ( viewer ) delete viewer;
  groupList.clear();
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
  
  writer.addElement("GroupViewer", attributes, String(0));

  
  if ( addSub == 0 ) {
    attributes.setAttribute( "name", group );
    attributes.setAttribute( "viewer", viewer );
    writer.addElement( "add", attributes, String(0) );
    attributes.clear();
  }  
  else if ( addSub == 1 ) {
    attributes.setAttribute( "name", group );
    attributes.setAttribute( "viewer", viewer );
    writer.addElement( "subtract", attributes, String(0) );
    attributes.clear();
  }
  else {
    // If addSub isn't set, we send a list of all the groups
    for (vector<struct groupListItem*>::iterator i = groupList.begin();
	 i != groupList.end();
	 i++) {
      attributes.setAttribute("name",    (*i)->group);
      attributes.setAttribute("viewer", (*i)->viewer);
      writer.addElement( "group", attributes, String(0));
      attributes.clear();
    }
  }
  
  //output = writer.writeOutputData();
  mkOutput( writer.writeOutputData() );
  finished = true;
  
}

void
GroupViewer::addGroup( const char * group, const char * viewer ) {

  if ( group == NULL || viewer == NULL )
    return;
  
  char * _group = strdup( group );
  char * _viewer = strdup( viewer );
  groupList.push_back( new groupListItem( _group, _viewer ) );
}

void
GroupViewer::groupAdded( const char * group, const char * viewer ) {
  if ( !group || !viewer ) return;
  
  if ( this->group ) delete this->group;
  if ( this->viewer ) delete this->viewer;

  this->group = strdup( group );
  this->viewer = strdup( viewer );
  addSub = 0; // ADD
}

  
void
GroupViewer::groupSubtracted( const char * group, const char * viewer ) {
  if ( !group || !viewer ) return;
  
  if ( this->group ) delete this->group;
  if ( this->viewer ) delete this->viewer;

  this->group = strdup( group );
  this->viewer = strdup( viewer );
  addSub = 1; // SUB
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

  element = reader.nextElement(); // request
  if ( element == NULL )
    return NULL;
  
  g = scinew GroupViewer( true );

  return g;
}

}
}
//
// $Log$
// Revision 1.1  2003/07/22 15:46:18  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:03  simpson
// Adding CollabVis files/dirs
//
