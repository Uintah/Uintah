
//#include <Core/Util/NotFinished.h>
#include "LevelIterator.h"
#include "Brick.h"
#include <Core/Malloc/Allocator.h>
#include <iostream>

namespace Kurt {

using namespace SCIRun;

LevelIterator::LevelIterator(const GLTexture3D* tex, Ray view,
				 Point control, int level):
  GLTextureIterator( tex, view, control), level( level )
{

  const Octree< Brick* >* node = tex->getBonTree();
  // SCICore::Malloc::AuditAllocator(SCICore::Malloc::default_allocator);
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
      if( (*node)()->level() == level)
	break;
    } while (node->type() == Octree<Brick* >::PARENT);
    
    next = (*node)();
  }
  if( next == 0 ) done = true;
}
  
Brick*
LevelIterator::Start()
{
    return next;
}

Brick*
LevelIterator::Next()
{
  // Get the last iterator
    if( !done )
      SetNext();
    return next;

}

bool
LevelIterator::isDone()
{
  return done;
}

void 
LevelIterator::SetNext()
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
    if( (*node)()->level() == level)
      break;
    order.push_back( traversal( node ));
    path.push_back( node );
    child = order.back()->front();
    order.back()->pop_front();
    node = (*node)[child];
  }
  next = (*node)();
}

} //Kurt
