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

//    File   : ClipField.h
//    Author : Michael Callahan
//    Date   : August 2001

#if !defined(ClipField_h)
#define ClipField_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Datatypes/Clipper.h>
#include <sci_hash_map.h>
#include <algorithm>

namespace SCIRun {


class GuiInterface;

class ClipByFunctionAlgo : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(ProgressReporter *reporter,
			      FieldHandle fieldh,
			      int clipmode) = 0;

  virtual string identify() = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    string clipfunction,
					    int hashoffset);
};


template <class FIELD>
class ClipByFunctionAlgoT : public ClipByFunctionAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter,
			      FieldHandle fieldh,
			      int clipmode);

  virtual bool vinside_p(double x, double y, double z,
			 typename FIELD::value_type v)
  {
    return false;
  };
};


template <class FIELD>
FieldHandle
ClipByFunctionAlgoT<FIELD>::execute(ProgressReporter *mod,
				    FieldHandle fieldh,
				    int clipmode)
{
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  typename FIELD::mesh_type *mesh =
    dynamic_cast<typename FIELD::mesh_type *>(fieldh->mesh().get_rep());
  typename FIELD::mesh_type *clipped = scinew typename FIELD::mesh_type();
  *(PropertyManager *)clipped = *(PropertyManager *)mesh;

#ifdef HAVE_HASH_MAP
  typedef hash_map<unsigned int,
    typename FIELD::mesh_type::Node::index_type,
    hash<unsigned int>,
    equal_to<unsigned int> > hash_type;
#else
  typedef map<unsigned int,
    typename FIELD::mesh_type::Node::index_type,
    equal_to<unsigned int> > hash_type;
#endif

  hash_type nodemap;

  vector<typename FIELD::mesh_type::Elem::index_type> elemmap;

  const bool elemdata_valid =
    field->data_at_type_description()->get_name() ==
    get_type_description((typename FIELD::mesh_type::Elem *)0)->get_name();

  typename FIELD::mesh_type::Elem::iterator bi, ei;
  mesh->begin(bi); mesh->end(ei);
  while (bi != ei)
  {
    bool inside = false;
    if (clipmode == 0)
    {
      Point p;
      mesh->get_center(p, *bi);
      typename FIELD::value_type v(0);
      if (elemdata_valid) { field->value(v, *bi); }
      inside = vinside_p(p.x(), p.y(), p.z(), v);
    }
    else if (clipmode > 0)
    {
      typename FIELD::mesh_type::Node::array_type onodes;
      mesh->get_nodes(onodes, *bi);

      inside = false;
      int counter = 0;
      for (unsigned int i = 0; i < onodes.size(); i++)
      {
	Point p;
	mesh->get_center(p, onodes[i]);
	typename FIELD::value_type v(0);
	if (field->data_at() == Field::NODE) { field->value(v, onodes[i]); }
	if (vinside_p(p.x(), p.y(), p.z(), v))
	{
	  counter++;
	  if (counter >= clipmode)
	  {
	    inside = true;
	    break;
	  }
	}
      }
    }
    else
    {
      typename FIELD::mesh_type::Node::array_type onodes;
      mesh->get_nodes(onodes, *bi);
      inside = true;
      for (unsigned int i = 0; i < onodes.size(); i++)
      {
	Point p;
	mesh->get_center(p, onodes[i]);
	typename FIELD::value_type v(0);
	if (field->data_at() == Field::NODE) { field->value(v, onodes[i]); }
	if (!vinside_p(p.x(), p.y(), p.z(), v))
	{
	  inside = false;
	  break;
	}
      }
    }

    if (inside)
    {
      typename FIELD::mesh_type::Node::array_type onodes;
      mesh->get_nodes(onodes, *bi);

      // Add this element to the new mesh.
      typename FIELD::mesh_type::Node::array_type nnodes(onodes.size());

      for (unsigned int i = 0; i<onodes.size(); i++)
      {
	if (nodemap.find((unsigned int)onodes[i]) == nodemap.end())
	{
	  Point np;
	  mesh->get_center(np, onodes[i]);
	  nodemap[(unsigned int)onodes[i]] = clipped->add_point(np);
	}
	nnodes[i] = nodemap[(unsigned int)onodes[i]];
      }

      clipped->add_elem(nnodes);
      elemmap.push_back(*bi); // Assumes elements always added to end.
    }
    
    ++bi;
  }

  FIELD *ofield = scinew FIELD(clipped, fieldh->data_at());
  *(PropertyManager *)ofield = *(PropertyManager *)(fieldh.get_rep());

  if (fieldh->data_at() == Field::NODE)
  {
    FIELD *field = dynamic_cast<FIELD *>(fieldh.get_rep());
    typename hash_type::iterator hitr = nodemap.begin();

    while (hitr != nodemap.end())
    {
      typename FIELD::value_type val;
      field->value(val, (typename FIELD::mesh_type::Node::index_type)((*hitr).first));
      ofield->set_value(val, (typename FIELD::mesh_type::Node::index_type)((*hitr).second));

      ++hitr;
    }
  }
  else if (fieldh->data_at_type_description()->get_name() ==
	   get_type_description((typename FIELD::mesh_type::Elem *)0)->get_name())
  {
    FIELD *field = dynamic_cast<FIELD *>(fieldh.get_rep());
    for (unsigned int i=0; i < elemmap.size(); i++)
    {
      typename FIELD::value_type val;
      field->value(val,
		   (typename FIELD::mesh_type::Elem::index_type)elemmap[i]);
      ofield->set_value(val, (typename FIELD::mesh_type::Elem::index_type)i);
    }
  }
  else
  {
    mod->warning("Unable to copy data at this field data location.");
  }

  return ofield;
}



} // end namespace SCIRun

#endif // ClipField_h
