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


#ifndef MODELCREATION_CORE_FIELDS_FIELDDATAELEMTONODE_H
#define MODELCREATION_CORE_FIELDS_FIELDDATAELEMTONODE_H 1

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/DynamicCompilation.h>
#include <Core/Util/ProgressReporter.h>


// Basis classes
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Basis/Bases.h>

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Mesh.h>


#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <algorithm>
#include <sgi_stl_warnings_on.h>

namespace ModelCreation {

using namespace SCIRun;
using namespace std;

class FieldDataElemToNodeAlgo : public DynamicAlgoBase
{
public:

  virtual bool execute(ProgressReporter *reporter,
                              FieldHandle input,
                              FieldHandle& output,
                              std::string method) = 0;

  static CompileInfoHandle get_compile_info(FieldHandle input);                                   
};


template<class FIELD, class OFIELD>
class FieldDataElemToNodeAlgoT : public FieldDataElemToNodeAlgo
{
public:
  virtual bool execute(ProgressReporter *reporter,
                              FieldHandle input,
                              FieldHandle& output,
                              std::string method);                          
};


template<class FIELD, class OFIELD>
bool FieldDataElemToNodeAlgoT<FIELD,OFIELD>::execute(ProgressReporter *reporter,
                              FieldHandle input,
                              FieldHandle& output,
                              std::string method)
{     

  output = 0;
  
  FIELD* field = dynamic_cast<FIELD* >(input.get_rep());
  if (field == 0)
  {
    reporter->error("FieldDataElemToNode: Object is not valid");
    return(false);
  }

  if (field->basis_order() > 0)
  {
     reporter->warning("FieldDataElemToNode: Data is already located at nodes");
     output = input;
     return(true);   
  }
  
  if (field->basis_order() != 0)
  {
     reporter->error("FieldDataElemToNode: Data is not located at elements");
     return(false);   
  }


  // Create the field with the new mesh and data location.
  OFIELD *ofield = scinew OFIELD(field->get_typed_mesh());
  ofield->resize_fdata();
  
  typename FIELD::mesh_handle_type mesh = field->get_typed_mesh();

  if (mesh->dimensionality() == 1)
  {
    typename FIELD::mesh_type::Edge::array_type edgearray;
    typename FIELD::mesh_type::Node::iterator it, eit;

    mesh->synchronize(SCIRun::Mesh::EDGES_E | SCIRun::Mesh::NODE_NEIGHBORS_E);
    mesh->begin(it);
    mesh->end(eit);

    if ((method == "Interpolate")||(method == "Average"))
    {
      while (it != eit)
      {
        mesh->get_edges(edgearray, *(it));
        int nsize = edgearray.size();
        typename FIELD::value_type val = 0.0;
        for (size_t p = 0; p < nsize; p++)
        {
          val = val + field->value(edgearray[p]);
        }
        val = static_cast<typename FIELD::value_type>(val*(1.0/static_cast<double>(nsize)));
        ofield->set_value(val,*(it));
        ++it;
      }
    }
    
    if (method == "Max")
    {
      while (it != eit)
      {
        mesh->get_edges(edgearray, *(it));
        int nsize = edgearray.size();
        typename FIELD::value_type val = 0.0;
        typename FIELD::value_type tval = 0.0;
        if (nsize > 0)
        {
          val = field->value(edgearray[0]);
          for (size_t p = 1; p < nsize; p++)
          {
            tval = field->value(edgearray[p]);
            if (tval > val) val = tval;
          }
        }
        ofield->set_value(val,*(it));
        ++it;
      }
    }
    
    if (method == "Min")
    {
      while (it != eit)
      {
        mesh->get_edges(edgearray, *it);
        int nsize = edgearray.size();
        typename FIELD::value_type val = 0.0;
        typename FIELD::value_type tval = 0.0;
        if (nsize > 0)
        {
          val = field->value(edgearray[0]);
          for (size_t p = 1; p < nsize; p++)
          {
            tval = field->value(edgearray[p]);
            if (tval < val) val = tval;
          }
        }
        ofield->set_value(val,*(it));
        ++it;
      }    
    }

    if (method == "Sum")
    {
      while (it != eit)
      {
        mesh->get_edges(edgearray, *(it));
        int nsize = edgearray.size();
        typename FIELD::value_type val = 0.0;
        for (size_t p = 0; p < nsize; p++)
        {
          val += field->value(edgearray[p]);
        }
        ofield->set_value(val,*(it));
        ++it;
      }
    }

    if (method == "Median")
    {
      while (it != eit)
      {
        mesh->get_edges(edgearray, *(it));
        int nsize = edgearray.size();
        std::vector<typename FIELD::value_type> valarray(nsize);
        for (size_t p = 0; p < nsize; p++)
        {
          valarray[p] = field->value(edgearray[p]);
        }
        sort(valarray.begin(),valarray.end());
        int idx = static_cast<int>((valarray.size()/2));
        ofield->set_value(valarray[idx],*(it));
        ++it;
      }
    }
  }

  if (mesh->dimensionality() == 2)
  {
    typename FIELD::mesh_type::Face::array_type facearray;
    typename FIELD::mesh_type::Node::iterator it, eit;

    mesh->synchronize(SCIRun::Mesh::FACES_E | SCIRun::Mesh::NODE_NEIGHBORS_E);
    mesh->begin(it);
    mesh->end(eit);

    if ((method == "Interpolate")||(method == "Average"))
    {
      while (it != eit)
      {
        mesh->get_faces(facearray, *(it));
        int nsize = facearray.size();
        typename FIELD::value_type val = 0.0;
        for (size_t p = 0; p < nsize; p++)
        {
          val = val + field->value(facearray[p]);
        }
        val = static_cast<typename FIELD::value_type>(val*(1.0/static_cast<double>(nsize)));
        ofield->set_value(val,*(it));
        ++it;
      }
    }
    
    if (method == "Max")
    {
      while (it != eit)
      {
        mesh->get_faces(facearray, *(it));
        int nsize = facearray.size();
        typename FIELD::value_type val = 0.0;
        typename FIELD::value_type tval = 0.0;
        if (nsize > 0)
        {
          val = field->value(facearray[0]);
          for (size_t p = 1; p < nsize; p++)
          {
            tval = field->value(facearray[p]);
            if (tval > val) val = tval;
          }
        }
        ofield->set_value(val,*(it));
        ++it;
      }
    }
    
    if (method == "Min")
    {
      while (it != eit)
      {
        mesh->get_faces(facearray, *it);
        int nsize = facearray.size();
        typename FIELD::value_type val = 0.0;
        typename FIELD::value_type tval = 0.0;
        if (nsize > 0)
        {
          val = field->value(facearray[0]);
          for (size_t p = 1; p < nsize; p++)
          {
            tval = field->value(facearray[p]);
            if (tval < val) val = tval;
          }
        }
        ofield->set_value(val,*(it));
        ++it;
      }    
    }

    if (method == "Sum")
    {
      while (it != eit)
      {
        mesh->get_faces(facearray, *(it));
        int nsize = facearray.size();
        typename FIELD::value_type val = 0.0;
        for (size_t p = 0; p < nsize; p++)
        {
          val += field->value(facearray[p]);
        }
        ofield->set_value(val,*(it));
        ++it;
      }
    }

    if (method == "Median")
    {
      while (it != eit)
      {
        mesh->get_faces(facearray, *(it));
        int nsize = facearray.size();
        std::vector<typename FIELD::value_type> valarray(nsize);
        for (size_t p = 0; p < nsize; p++)
        {
          valarray[p] = field->value(facearray[p]);
        }
        sort(valarray.begin(),valarray.end());
        int idx = static_cast<int>((valarray.size()/2));
        ofield->set_value(valarray[idx],*(it));
        ++it;
      }
    }
  }

  if (mesh->dimensionality() == 3)
  {
    typename FIELD::mesh_type::Cell::array_type cellarray;
    typename FIELD::mesh_type::Node::iterator it, eit;

    mesh->synchronize(SCIRun::Mesh::CELLS_E | SCIRun::Mesh::NODE_NEIGHBORS_E);
    mesh->begin(it);
    mesh->end(eit);

    if ((method == "Interpolate")||(method == "Average"))
    {
      while (it != eit)
      {
        mesh->get_cells(cellarray, *(it));
        int nsize = cellarray.size();
        typename FIELD::value_type val = 0.0;
        for (size_t p = 0; p < nsize; p++)
        {
          val = val + field->value(cellarray[p]);
        }
        val = static_cast<typename FIELD::value_type>(val*(1.0/static_cast<double>(nsize)));
        ofield->set_value(val,*(it));
        ++it;
      }
    }
    
    if (method == "Max")
    {
      while (it != eit)
      {
        mesh->get_cells(cellarray, *(it));
        int nsize = cellarray.size();
        typename FIELD::value_type val = 0.0;
        typename FIELD::value_type tval = 0.0;
        if (nsize > 0)
        {
          val = field->value(cellarray[0]);
          for (size_t p = 1; p < nsize; p++)
          {
            tval = field->value(cellarray[p]);
            if (tval > val) val = tval;
          }
        }
        ofield->set_value(val,*(it));
        ++it;
      }
    }
    
    if (method == "Min")
    {
      while (it != eit)
      {
        mesh->get_cells(cellarray, *it);
        int nsize = cellarray.size();
        typename FIELD::value_type val = 0.0;
        typename FIELD::value_type tval = 0.0;
        if (nsize > 0)
        {
          val = field->value(cellarray[0]);
          for (size_t p = 1; p < nsize; p++)
          {
            tval = field->value(cellarray[p]);
            if (tval < val) val = tval;
          }
        }
        ofield->set_value(val,*(it));
        ++it;
      }    
    }

    if (method == "Sum")
    {
      while (it != eit)
      {
        mesh->get_cells(cellarray, *(it));
        int nsize = cellarray.size();
        typename FIELD::value_type val = 0.0;
        for (size_t p = 0; p < nsize; p++)
        {
          val += field->value(cellarray[p]);
        }
        ofield->set_value(val,*(it));
        ++it;
      }
    }

    if (method == "Median")
    {
      while (it != eit)
      {
        mesh->get_cells(cellarray, *(it));
        int nsize = cellarray.size();
        std::vector<typename FIELD::value_type> valarray(nsize);
        for (size_t p = 0; p < nsize; p++)
        {
          valarray[p] = field->value(cellarray[p]);
        }
        sort(valarray.begin(),valarray.end());
        int idx = static_cast<int>((valarray.size()/2));
        ofield->set_value(valarray[idx],*(it));
        ++it;
      }
    }
  }

  output = dynamic_cast<SCIRun::Field *>(ofield);
  return(true);
}


} // namespace ModelCreation

#endif
