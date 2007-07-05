#include <Rendering/RenderGroup.h>

namespace SemotusVisum {
namespace Rendering {

list<RenderGroup *>
RenderGroup::renderGroups;

RenderGroup::RenderGroup( char * name ) : renderer( NULL ), compressor( NULL ),
					  multicast( NULL ) {
  if ( name != NULL )
    this->name = strdup( name );
  renderGroups.push_front( this );
}


RenderGroup::~RenderGroup() {
  if ( name )
    delete name;
  clients.clear();
  renderGroups.remove( this );
}


RenderGroup* 
RenderGroup::getRenderGroup( const char * clientName ) {
  list<RenderGroup*>::iterator i;
  list<char *> clients;
  list<char *>::iterator j;
  
  for ( i = renderGroups.begin(); i != renderGroups.end(); i++ ) {
    clients = (*i)->getClients();
    for ( j = clients.begin(); j != clients.end(); j++ ) {
      if ( !strcmp( clientName, *j ) )
	return *i;
    }
  }

  return NULL;
}

}
}
