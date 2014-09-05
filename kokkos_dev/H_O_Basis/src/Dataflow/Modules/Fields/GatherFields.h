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
#include <sstream>
#include <iomanip>
#include <sci_hash_map.h>

namespace SCIRun {

class GatherPointsAlgo : public DynamicAlgoBase
{
public:
  typedef PointCloudMesh<ConstantBasis<Point> > PCMesh;

  virtual void execute(MeshHandle src, PCMesh::handle_type pcmH) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *msrc);
};


template <class MESH>
class GatherPointsAlgoT : public GatherPointsAlgo
{
public:
  //! virtual interface. 
  virtual void execute(MeshHandle src, PCMesh::handle_type pcmH);
};


template <class MESH>
void 
GatherPointsAlgoT<MESH>::execute(MeshHandle mesh_h, PCMesh::handle_type pcmH)
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
  virtual FieldHandle execute(vector<FieldHandle> &fields,
                              int out_basis, bool cdata, int precision) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftype);
};


template <class FIELD>
class GatherFieldsAlgoT : public GatherFieldsAlgo
{
  struct str_hasher
  {
    size_t operator()(const string s) const
    {
      hash<const char*> H;
      return H(s.c_str());
    }
  };
  struct eqstr
  {
    bool operator()(const string s1, const string s2) const
    {
      return s1 == s2;
    }
  };
  typedef hash_map<string, unsigned, str_hasher, eqstr> points_ht;
  points_ht pnts_table_;

  struct equint
  {
    bool operator()(const unsigned s1, const unsigned s2) const
    {
      return s1 == s2;
    }
  };
  typedef hash_map<unsigned, unsigned, hash<unsigned>, equint> idx_ht;
  idx_ht node_table_;


protected:
  FieldHandle append_fields(vector<FIELD *> fields,
                            int out_basis, bool cdata, int precision);

public:
  //! virtual interface. 
  virtual FieldHandle execute(vector<FieldHandle> &fields,
                              int out_basis, bool cdata, int precision);
};

string
hash_str(const Point &pnt, int prec) 
{
  ostringstream str;
  str << setiosflags(ios::scientific);
  str << setprecision(prec);
  str << pnt.x() << pnt.y() << pnt.z(); 
  string s = str.str();
  //  cerr << prec << " : " << s << std::endl;

  return s;
}

template <class FIELD>
FieldHandle
GatherFieldsAlgoT<FIELD>::append_fields(vector<FIELD *> fields,
                                        int out_basis, bool cdata, 
					int precision)
{
  typename FIELD::mesh_type *omesh = scinew typename FIELD::mesh_type();
  pnts_table_.clear();
  unsigned int offset = 0;
  unsigned int i;
  for (i=0; i < fields.size(); i++)
  {
    node_table_.clear();
    typename FIELD::mesh_handle_type imesh = fields[i]->get_typed_mesh();
    typename FIELD::mesh_type::Node::iterator nitr, nitr_end;
    imesh->begin(nitr);
    imesh->end(nitr_end);
    unsigned int input_field_node_idx = 0;
    while (nitr != nitr_end)
    {
      Point p;
      imesh->get_center(p, *nitr);
      //hash p
      // if it is unique map index to index
      // else map index to previously hashed index
      string p_str = hash_str(p, precision);
      typename points_ht::iterator ins = pnts_table_.find(p_str);
      unsigned out_node_idx;
      if(ins == pnts_table_.end()) {
	// p was not in the table.
	out_node_idx = omesh->add_point(p);
	pnts_table_[p_str] = out_node_idx;

      } else {
	// p already existed in the table
	out_node_idx = (*ins).second;
      }
      node_table_[input_field_node_idx] = out_node_idx;
      ++nitr; ++input_field_node_idx;
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
	// set the indeces to the matching output node indeces.
	typename idx_ht::iterator nt_iter = 
	  node_table_.find((unsigned int)nodes[j]);
	nodes[j] = (*nt_iter).second;
      }
      omesh->add_elem(nodes);
      ++eitr;
    }
    
    typename FIELD::mesh_type::Node::size_type size;
    imesh->size(size);
  }

  FIELD *ofield = scinew FIELD(omesh, out_basis);
  if (cdata)
  {
    if (out_basis == 0)
    {
      offset = 0;
      for (i=0; i < fields.size(); i++)
      {
        typename FIELD::mesh_handle_type imesh = fields[i]->get_typed_mesh();
        typename FIELD::mesh_type::Elem::iterator itr, itr_end;
        imesh->begin(itr);
        imesh->end(itr_end);
        while (itr != itr_end)
        {
          typename FIELD::value_type val;
          fields[i]->value(val, *itr);
          typename FIELD::mesh_type::Elem::index_type
            new_index(((unsigned int)(*itr)) + offset);
          ofield->set_value(val, new_index);
          ++itr;
        }

        typename FIELD::mesh_type::Elem::size_type size;
        imesh->size(size);
        offset += (unsigned int)size;
      }
    }
    if (out_basis == 1)
    {
      for (i=0; i < fields.size(); i++)
      {
        typename FIELD::mesh_handle_type imesh = fields[i]->get_typed_mesh();
        typename FIELD::mesh_type::Node::iterator nitr, nitr_end;
        imesh->begin(nitr);
        imesh->end(nitr_end);
        while (nitr != nitr_end)
        {
	  // Note: that for duplicated points it is assumed that these input
	  // nodes have the same data associated with them.  There is no 
          // averaging, just last one written wins.
          typename FIELD::value_type val;
          fields[i]->value(val, *nitr);

	  typename idx_ht::iterator nt_iter = 
	    node_table_.find((unsigned int)*nitr);
          typename FIELD::mesh_type::Node::index_type 
	    new_index((*nt_iter).second);
          ofield->set_value(val, new_index);
          ++nitr;
        }

        typename FIELD::mesh_type::Node::size_type size;
        imesh->size(size);
      }
    }
    // basis == -1, no data to copy.
  }
  return ofield;
}


template <class FIELD>
FieldHandle
GatherFieldsAlgoT<FIELD>::execute(vector<FieldHandle> &fields_h,
                                  int out_basis, bool cdata, int p)
{
  vector<FIELD *> fields;
  for (unsigned int i = 0; i < fields_h.size(); i++)
  {
    fields.push_back((FIELD*)(fields_h[i].get_rep()));
  }
  
  return append_fields(fields, out_basis, cdata, p);
}

} // End namespace SCIRun

#endif
