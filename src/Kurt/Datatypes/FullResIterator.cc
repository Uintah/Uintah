//#include <SCICore/Utils/NotFinished.h>
#include "FullResIterator.h"


namespace Kurt {
namespace Datatypes {

using SCICore::Geometry::Point;



FullResIterator::FullResIterator(const GLTexture3D* tex, Ray view,
				 Point control):
    GLTextureIterator( tex, view, control)
{
  const Octree< Brick* >* node = (tex->bontree);
  if ( tex->depth() == 0 ){
    next = (*node)();
  } else {
    int child;
    do {
      order.push_back( traversal( node ));
      path.push_back( node );
      child = order.back()->front();
      order.back()->pop_front();
      node = (*node)[child];
    } while (node->type() == Octree<Brick* >::PARENT);
    
    next = (*node)();
  }
}
  
Brick*
FullResIterator::Start()
{
    return next;
}

Brick*
FullResIterator::Next()
{
  // Get the last iterator
    if( !done )
      SetNext();
    return next;

}

bool
FullResIterator::isDone()
{
  return done;
}

void 
FullResIterator::SetNext()
{ 
  while( path.size() != 0 && 
	 !order.back()->size() ){
    path.pop_back();
    order.pop_back();
  }

  if( path.size() == 0 ){
    done = true;
    return;
  }

  int child;
  const Octree< Brick* >* node = path.back();
  child = order.back()->front();
  order.back()->pop_front();
  node = (*node)[child];
  
  while( node->type() == Octree<Brick* >::PARENT ){
    order.push_back( traversal( node ));
    path.push_back( node );
    child = order.back()->front();
    order.back()->pop_front();
    node = (*node)[child];
  }
  next = (*node)();
}

} //Datatypes
} //Kurt

