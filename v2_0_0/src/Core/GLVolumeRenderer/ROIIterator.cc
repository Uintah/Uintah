/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

#include <Core/GLVolumeRenderer/ROIIterator.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Vector.h>
#include <Core/GLVolumeRenderer/Brick.h>
#include <iostream>

namespace SCIRun {




ROIIterator::ROIIterator(const GLTexture3D* tex, Ray view,
			 Point control):
    GLTextureIterator( tex, view, control )
{
  const Octree< Brick* >* node = (tex->bontree_);
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
      if( !b.inside( control )){
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
    if(!b.inside( control )){
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

} // End namespace SCIRun

