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


#include <Core/GLVolumeRenderer/GLTextureIterator.h>
#include <Core/GLVolumeRenderer/Brick.h>
#include <Core/Geometry/Transform.h>
#include <iostream>

namespace SCIRun {


int GLTextureIterator::traversalTable[27][8] = { {7,3,5,6,1,2,4,0},
					  {6,7,2,3,4,5,0,1},
					  {6,2,4,7,0,3,5,1},
					  {5,7,1,3,4,6,0,2},
					  {4,5,6,7,0,1,2,3},
					  {4,6,0,2,5,7,1,3},
					  {5,1,4,7,0,3,6,2},
					  {4,5,0,1,6,7,2,3},
					  {4,0,5,6,1,2,7,3},
					  {3,7,1,2,5,6,0,4},
					  {2,3,6,7,0,1,4,5},
					  {2,6,0,3,4,7,1,5},
					  {1,3,5,7,0,2,4,6},
					  {0,1,2,3,4,5,6,7},
					  {0,2,4,6,1,3,5,7},
					  {1,5,0,3,4,7,2,6},
					  {0,1,4,5,2,3,6,7},
					  {0,4,1,2,5,6,3,7},
					  {3,1,2,7,0,5,6,4},
					  {2,3,0,1,6,7,4,5},
					  {2,0,3,6,1,4,7,5},
					  {1,3,0,2,5,7,4,6},
					  {0,1,2,3,4,5,6,7},
					  {0,2,1,3,4,6,5,7},
					  {1,0,3,5,2,4,7,6},
					  {0,1,2,3,4,5,6,7},
					  {0,1,2,4,3,5,6,7}};



GLTextureIterator::GLTextureIterator(const GLTexture3D* tex,
				     Ray view,
				     Point c )
  : view(view),
    control(c),
    tex(tex),
    done(false)
{
  Transform t = tex->get_field_transform();
  control =  t.unproject(c);
}


GLTextureIterator::~GLTextureIterator()
{
}

deque<int>* 
GLTextureIterator::traversal(const Octree<Brick*>* n)
{
  const Octree<Brick*>& node = *n;
  const Octree<Brick*>& child = *(node[0]);


  int *traversal;
  int traversalIndex, x, y, z;

  Point min, mid;
  mid = child()->bbox().max();
  min = child()->bbox().min();

  if( view.origin().x() < mid.x()) x = 0;
  else if( view.origin().x() == mid.x()) x = 1;
  else x = 2;
  if( view.origin().y() < mid.y()) y = 0;
  else if( view.origin().y() == mid.y()) y = 1;
  else y = 2;
  if( view.origin().z() < mid.z()) z = 0;
  else if( view.origin().z() == mid.z()) z = 1;
  else z = 2;
    
  traversalIndex = 9*x + 3*y + z;

  traversal = traversalTable[ traversalIndex ];

  deque< int > *q = scinew deque< int >;
  
  //  std::cerr<<"Child order ";
  for( int i = 0; i < 8; i++){
    if( node[traversal[i]] && ((*(node[ traversal[i] ]))() != 0)){
      //      std::cerr<<traversal[i]<<" ";
      q->push_back( traversal[i] );
    }
  }
  //  std::cerr<<std::endl;
  return q;
}

} // End namespace SCIRun
