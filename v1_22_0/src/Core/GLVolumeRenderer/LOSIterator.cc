/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


#include <Core/GLVolumeRenderer/LOSIterator.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Vector.h>
#include <Core/GLVolumeRenderer/Brick.h>


namespace SCIRun {




LOSIterator::LOSIterator(const GLTexture3D* tex, Ray view,
			 Point control):
    GLTextureIterator( tex, view, control )
{
  Vector v = control - view.origin();
  Point p;
  const Octree< Brick* >* node = (tex->bontree_);
  if ( tex->depth() == 0 ){
    next = (*node)();
  } else {
    int child;
    BBox box((*node)()->bbox());
    if( !box.intersect( view.origin(), v, p )){
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
      if( !b.intersect( view.origin(), v, p )){
	break;
      }
    } while (node->type() == Octree<Brick* >::PARENT);
    
    next = (*node)();
  }

}


Brick*
LOSIterator::Start()
{
  return next;
}

Brick*
LOSIterator::Next()
{
  // Get the last iterator
    if( !done )
      SetNext();
    return next;
}

bool
LOSIterator::isDone()
{
  return done;
}

void 
LOSIterator::SetNext()
{ 
  Vector v = control - view.origin();
  Point p;

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
    if(!b.intersect( view.origin(), v, p )){
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

