#include <Message/Collaborate.h>

namespace SemotusVisum {
namespace Message {

using namespace XML;

Collaborate::Collaborate( bool request ) : request(request),
					   text( NULL ) {

}


Collaborate::~Collaborate() {
  delete text;
}

void 
Collaborate::finish() {

  if ( finished )
    return;
  
  // We have a special case here - we simply bounce the message to clients.
  mkOutput( text );
  finished = true;
}

void
Collaborate::switchID( const char * newID ) {
  if ( text == NULL )
    return;

  // First, find out how many occurances of "Local" we have
  int count = 0;
  int start = 0;
  std::cerr << "Finding locals..." << endl;
  while ( (start = nextLocal( start, text ) ) != -1 ) {
    std::cerr << "current position: " << start << endl;
    start += strlen("local");
    count++;
  }

  std::cerr << "We have " << count << " instances of local" << endl;
  
  // Now get a new data buffer that can accomodate the increased size.
  int size = strlen(text) + 1 + count*(strlen(newID) - strlen("local"));
  char * newData = scinew char[ size ];
  memset( newData, 0, size );
  
  // For each instance of "Local" in the text, replace that with
  // the new ID.
  start = 0;
  int next;
  char * nextData = newData;
  while ( (next = nextLocal( start, text ) ) != -1 ) {
    memcpy( nextData, text + start, next-start );
    nextData += next-start;
    memcpy( nextData, newID, strlen(newID) );
    nextData += strlen(newID);

    start = next + strlen("local");
  }
  memcpy( nextData, text+start, strlen(text) - start );

  delete[] text;
  text = newData;
}


Collaborate *  
Collaborate::mkCollaborate( void * data ) {

  Collaborate * c = NULL;
  
  char * text = XMLI::getChar( (char *)data );

  c = new Collaborate( true );
  c->setText( text );
  
  return c;
}

int
Collaborate::nextLocal( int start, const char * text ) {
  char * local = "local";
  int okay = 0;
  for ( int i = start; i < strlen(text); i++ ) {
    if ( tolower(text[i]) == local[okay] )
      okay++;
    else 
      okay = 0;
    if ( okay == 5 )
      return i - 4;
  }
  return -1;
    
}

}
}
//
// $Log$
// Revision 1.1  2003/07/22 15:46:17  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:01  simpson
// Adding CollabVis files/dirs
//
// Revision 1.2  2001/09/25 14:44:58  luke
// Got collaboration working
//
// Revision 1.1  2001/09/23 02:24:11  luke
// Added collaborate message
//

