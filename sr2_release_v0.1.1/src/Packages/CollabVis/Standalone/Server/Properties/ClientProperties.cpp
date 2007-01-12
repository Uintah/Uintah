#include <Properties/ClientProperties.h>
#include <Properties/ServerProperties.h>
#include <Malloc/Allocator.h>

namespace SemotusVisum {
namespace Properties {

using namespace Logging;
using namespace XML;
using namespace std;

map<char *, ClientProperties*>
ClientProperties::clientList;

ClientProperties::ClientProperties() : name( NULL ) {
  //ClientProperties::clientList.insert( make_pair( NULL, this ) );
}

ClientProperties::ClientProperties( const char * name ) {
  this->name = strdup( name );
  ClientProperties::clientList.insert( make_pair( this->name, this ) );
}
				       
ClientProperties::~ClientProperties() {
  
  /* Clear the lists - this will delete the objects */
  
  imageFormats.clear();
  availableRenderers.clear();
  compressors.clear();

  /* Remove ourselves from the global list */
  if ( name ) {
    ClientProperties::clientList.erase( name );
    delete name;
  }
}

bool
ClientProperties::getFormats( Handshake &handshake, ClientProperties &c ) {

  char buffer[ 1000 ];
  list<imageFormat>&     i = handshake.getImageFormats();
  list<renderInfo>&      j = handshake.getViewMethods();
  list<compressionInfo>& k = handshake.getCompress();

  Log::log( Logging::MESSAGE, "Setting Client Properties" );
  
  // Get the image formats
  for (list<imageFormat>::iterator q = i.begin();
       q != i.end();
       q++) {
    snprintf( buffer, 1000, "Adding image format %s", q->name );
    Log::log( Logging::MESSAGE, buffer ); 
    c.addImageFormat( *q );
  }
    
  // Get the view methods
  for (list<renderInfo>::iterator q = j.begin();
       q != j.end();
       q++){
    snprintf( buffer, 1000,
	      "Adding rendering method %s, version %s",
	      q->name,
	      q->version );
    Log::log( Logging::MESSAGE, buffer ); 
    c.addRenderInfo( *q );
  }
  
  // Get the compressors
  for (list<compressionInfo>::iterator q = k.begin();
       q != k.end();
       q++) {
    snprintf( buffer, 1000, "Adding compression format %s", q->name );
    Log::log( Logging::MESSAGE, buffer ); 
    c.addCompressor( *q );
  }

  return true;
}


void
ClientProperties::addImageFormat( const imageFormat& format )  {

  imageFormats.push_front( *( scinew imageFormat(format) ) );
  
}

void     
ClientProperties::addRenderInfo( const renderInfo& info ) {

  availableRenderers.push_front( *( scinew renderInfo(info) ) );
}

void     
ClientProperties::addCompressor( const compressionInfo& info ) {

  compressors.push_front( *( scinew compressionInfo(info) ) );
}

bool     
ClientProperties::validImageFormat( const imageFormat& format ) const {

  for ( imageList::const_iterator i = imageFormats.begin();
	i != imageFormats.end();
	i++ )

    if ( format == *i ) return true;

  return false;
}


bool     
ClientProperties::validRenderer( const renderInfo& info ) const {

  for ( renderList::const_iterator i = availableRenderers.begin();
	i != availableRenderers.end();
	i++ )

    if ( info == *i ) return true;

  return false;
}

bool     
ClientProperties::validCompressor( const compressionInfo& info ) const {
  
  for ( compressionList::const_iterator i = compressors.begin();
	i != compressors.end();
	i++ )

    if ( info == *i ) return true;

  return false;
  
}

void
ClientProperties::printProperties() const {

  std::cerr << "Image Formats:" << endl;
  for ( imageList::const_iterator i = imageFormats.begin();
	i != imageFormats.end();
	i++ )
    std::cerr << "\t" << i->name << " " << i->type << " " << i->depth << endl;


  std::cerr << endl << "Renderers:" << endl;
  for ( renderList::const_iterator j = availableRenderers.begin();
	j != availableRenderers.end();
	j++ )
    std::cerr << "\t" << j->name << " " << j->version << endl;

  std::cerr << endl << "Compressors:" << endl;
  for ( compressionList::const_iterator k = compressors.begin();
	k != compressors.end();
	k++ )
    std::cerr << "\t" << k->name << " " << k->type << endl;
  
}

ClientProperties*
ClientProperties::getProperties( const char * clientName ) {
  map<char *, ClientProperties*>::const_iterator i;
  
  for (i = clientList.begin();
       i != clientList.end();
       i++)
    if ( !strcmp( clientName, i->first ))
      return i->second;

  return NULL;

}

}
}
//
// $Log$
// Revision 1.1  2003/07/22 15:46:34  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 22:23:02  simpson
// Adding CollabVis files/dirs
//
// Revision 1.6  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.5  2001/05/29 03:43:12  luke
// Merged in changed to allow code to compile/run on IRIX. Note that we have a problem with network byte order in the networking code....
//
// Revision 1.4  2001/05/21 22:00:45  luke
// Got basic set viewing method to work. Problems still with reading 0 data from net and ignoring outbound messages
//
// Revision 1.3  2001/05/11 20:53:40  luke
// Moved properties to messages from XML. Moved properties drivers to new location.
//
