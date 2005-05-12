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


//    File   : FieldCount.h
//    Author : McKay Davis
//    Date   : July 2002

#if !defined(FieldCount_h)
#define FieldCount_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Field.h>
#include <sstream>

namespace SCIRun {

class FieldCountAlgorithm : public DynamicAlgoBase
{
public:
  virtual void execute(MeshHandle src, int &num_nodes, int &num_elems) = 0;
  virtual string execute_node(MeshHandle src) = 0;
  virtual string execute_elem(MeshHandle src) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *msrc);
};


template <class MESH>
class FieldCountAlgorithmT : public FieldCountAlgorithm
{
public:
  //! virtual interface. 
  virtual void execute(MeshHandle src, int &num_nodes, int &num_elems);
  virtual string execute_node(MeshHandle src);
  virtual string execute_elem(MeshHandle src);
};


template <class MESH>
void 
FieldCountAlgorithmT<MESH>::execute(MeshHandle mesh_h, 
				   int &num_nodes, 
				   int &num_elems)
{
  MESH *mesh = dynamic_cast<MESH *>(mesh_h.get_rep());
  typename MESH::Node::size_type nnodes;
  typename MESH::Elem::size_type nelems;
  mesh->size(nnodes);
  mesh->size(nelems);
  num_nodes=nnodes;
  num_elems=nelems;
}


template <class MESH>
string
FieldCountAlgorithmT<MESH>::execute_node(MeshHandle mesh_h)
{
  MESH *mesh = dynamic_cast<MESH *>(mesh_h.get_rep());

  // Nodes
  typename MESH::Node::size_type nnodes;
  mesh->size(nnodes);
  std::ostringstream nodestr;
  nodestr << nnodes;
  return nodestr.str();
}


template <class MESH>
string
FieldCountAlgorithmT<MESH>::execute_elem(MeshHandle mesh_h)
{
  MESH *mesh = dynamic_cast<MESH *>(mesh_h.get_rep());

  // Elements
  typename MESH::Elem::size_type nelems;  
  mesh->size(nelems);
  std::ostringstream elemstr;
  elemstr << nelems;
  return elemstr.str();
}



} // end namespace SCIRun

#endif // FieldInfo_h
