#include "ROIIterator.h"
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Vector.h>
#include "Brick.h"
#include <iostream>

namespace Kurt {

using namespace SCIRun;

ROIIterator::ROIIterator(const GLTexture3D* tex, Ray view,
			 Point control):
    GLTextureIterator( tex, view, control )
{
  const Octree< Brick* >* node = tex->getBonTree();
  if ( tex->depth() == 0 ){
    next = (*node)();
  } else {
    BBox box((*node)()->bbox());
    int child;
    if( !box.inside( control )){
      next = (*node)();
      return;
    }
    do {
      order.push_back( traversal( node ));
      path.push_back( node );
      child = order.back()->front();
      order.back()->pop_front();
      node = (*node)[child];
      BBox b((*node)()->bbox());
      if( !b.inside( control ) &&
	(*node)()->level() == 1){
	break;
      }
    } while (node->type() == Octree<Brick* >::PARENT);
    
    next = (*node)();
  }

}


Brick*
ROIIterator::Start()
{
  return next;
}

Brick*
ROIIterator::Next()
{
    // Get the last iterator
    if( !done )
      SetNext();
    return next;
}

bool
ROIIterator::isDone()
{
 return done;
}

void 
ROIIterator::SetNext()
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
    BBox b( (*node)()->bbox() );
    if(!b.inside( control ) &&
	(*node)()->level() == 1){
      break;
    }
    order.push_back( traversal( node ));
    path.push_back( node );
    child = order.back()->front();
    order.back()->pop_front();
    node = (*node)[child];
  }
  next = (*node)();
}

} // End namespace Kurt
