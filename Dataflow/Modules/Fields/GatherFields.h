#ifndef SCIRun_GatherFields_H
#define SCIRun_GatherFields_H
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


#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/PointCloudMesh.h>


namespace SCIRun {

class GatherPointsAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(MeshHandle src, PointCloudMeshHandle pcmH) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *msrc);
};


template <class MESH>
class GatherPointsAlgoT : public GatherPointsAlgo
{
public:
  //! virtual interface. 
  virtual void execute(MeshHandle src, PointCloudMeshHandle pcmH);
};


template <class MESH>
void 
GatherPointsAlgoT<MESH>::execute(MeshHandle mesh_h, PointCloudMeshHandle pcmH)
{
  typedef typename MESH::Node::iterator node_iter_type;

  MESH *mesh = dynamic_cast<MESH *>(mesh_h.get_rep());

  node_iter_type ni; mesh->begin(ni);
  node_iter_type nie; mesh->end(nie);
  while (ni != nie)
  {
    Point p;
    mesh->get_center(p, *ni);
    pcmH->add_node(p);
    ++ni;
  }
}



class GatherFieldsAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(vector<FieldHandle> &fields, bool cdata) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftype);
};


template <class FIELD>
class GatherFieldsAlgoT : public GatherFieldsAlgo
{
protected:
  FieldHandle append_fields(vector<FIELD *> fields, bool cdata);

public:
  //! virtual interface. 
  virtual FieldHandle execute(vector<FieldHandle> &fields, bool cdata);
};


template <class FIELD>
FieldHandle
GatherFieldsAlgoT<FIELD>::append_fields(vector<FIELD *> fields, bool cdata)
{
  typename FIELD::mesh_type *omesh = scinew typename FIELD::mesh_type();

  unsigned int offset = 0;
  unsigned int i;
  for (i=0; i < fields.size(); i++)
  {
    typename FIELD::mesh_handle_type imesh = fields[i]->get_typed_mesh();
    typename FIELD::mesh_type::Node::iterator nitr, nitr_end;
    imesh->begin(nitr);
    imesh->end(nitr_end);
    while (nitr != nitr_end)
    {
      Point p;
      imesh->get_center(p, *nitr);
      omesh->add_point(p);
      ++nitr;
    }

    typename FIELD::mesh_type::Elem::iterator eitr, eitr_end;
    imesh->begin(eitr);
    imesh->end(eitr_end);
    while (eitr != eitr_end)
    {
      typename FIELD::mesh_type::Node::array_type nodes;
      imesh->get_nodes(nodes, *eitr);
      unsigned int j;
      for (j = 0; j < nodes.size(); j++)
      {
	nodes[j] = ((unsigned int)nodes[j]) + offset;
      }
      omesh->add_elem(nodes);
      ++eitr;
    }
    
    typename FIELD::mesh_type::Node::size_type size;
    imesh->size(size);
    offset += (unsigned int)size;
  }

  FIELD *ofield = scinew FIELD(omesh, 1);
  if (cdata)
  {
    offset = 0;
    for (i=0; i < fields.size(); i++)
    {
      typename FIELD::mesh_handle_type imesh = fields[i]->get_typed_mesh();
      typename FIELD::mesh_type::Node::iterator nitr, nitr_end;
      imesh->begin(nitr);
      imesh->end(nitr_end);
      while (nitr != nitr_end)
      {
	typename FIELD::value_type val;
	fields[i]->value(val, *nitr);
	typename FIELD::mesh_type::Node::index_type
	  new_index(((unsigned int)(*nitr)) + offset);
	ofield->set_value(val, new_index);
	++nitr;
      }

      typename FIELD::mesh_type::Node::size_type size;
      imesh->size(size);
      offset += (unsigned int)size;
    }
  }
  return ofield;
}


template <class FIELD>
FieldHandle
GatherFieldsAlgoT<FIELD>::execute(vector<FieldHandle> &fields_h, bool cdata)
{
  vector<FIELD *> fields;
  for (unsigned int i = 0; i < fields_h.size(); i++)
  {
    fields.push_back((FIELD*)(fields_h[i].get_rep()));
  }
  
  return append_fields(fields, cdata);
}

} // End namespace SCIRun

#endif
