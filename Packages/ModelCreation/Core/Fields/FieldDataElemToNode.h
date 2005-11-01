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

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/Field.h>

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/DynamicCompilation.h>
#include <Core/Util/ProgressReporter.h>

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Mesh.h>

// Basis classes
#include <Core/Basis/CrvLinearLgn.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Basis/PrismLinearLgn.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Basis/TriLinearLgn.h>

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


template<class FIELD>
class FieldDataElemToNodeAlgoT : public FieldDataElemToNodeAlgo
{
public:
  virtual bool execute(ProgressReporter *reporter,
                              FieldHandle input,
                              FieldHandle& output,
                              std::string method);                          
};


// Do i have the following operators available; < > == etc.
// If not I cannot do operations like max, min, and median

inline bool has_compare(char) { return(true); }
inline bool has_compare(unsigned char) { return(true); }
inline bool has_compare(short) { return(true); }
inline bool has_compare(unsigned short) { return(true); }
inline bool has_compare(int) { return(true); }
inline bool has_compare(unsigned int) { return(true); }
inline bool has_compare(float) { return(true); }
inline bool has_compare(double) { return(true); }
inline bool has_compare(SCIRun::Tensor) { return(false); }
inline bool has_compare(SCIRun::Vector) { return(false); }

inline void zero(char &c) { c = 0; }
inline void zero(unsigned char &c) { c = 0; }
inline void zero(short &c) { c = 0; }
inline void zero(unsigned short &c) { c = 0; }
inline void zero(int &c) { c = 0; }
inline void zero(unsigned int &c) { c = 0; }
inline void zero(double &c) { c = 0.0; }
inline void zero(float &c) { c = 0.0; }
inline void zero(SCIRun::Vector &c) { c = SCIRun::Vector(0.0,0.0,0.0); }
inline void zero(SCIRun::Tensor &c) { c = SCIRun::Tensor(0.0); }

inline void sort(std::vector<SCIRun::Vector>::iterator beg,std::vector<SCIRun::Vector>::iterator end) {return; } 
inline void sort(std::vector<SCIRun::Tensor>::iterator beg,std::vector<SCIRun::Tensor>::iterator end) {return; }  


template<class FIELD>
bool FieldDataElemToNodeAlgoT<FIELD>::execute(ProgressReporter *reporter,
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
  FIELD *ofield = scinew FIELD(field->get_typed_mesh(), 1);
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
        typename FIELD::value_type val;
        zero(val);
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
      typename FIELD::value_type test;
      if (has_compare(test))
      {
        while (it != eit)
        {
          mesh->get_edges(edgearray, *(it));
          int nsize = edgearray.size();
          typename FIELD::value_type val;
          zero(val);
          typename FIELD::value_type tval;
          zero(tval);
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
      else
      {
        reporter->error("Maximum has not been implemented for this type of data");
        return(false);
      }
    }
    
    if (method == "Min")
    {
      typename FIELD::value_type test;
      if (has_compare(test))
      {
        while (it != eit)
        {
          mesh->get_edges(edgearray, *it);
          int nsize = edgearray.size();
          typename FIELD::value_type val;
          zero(val);
          typename FIELD::value_type tval;
          zero(tval);
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
      else
      {
        reporter->error("Maximum has not been implemented for this type of data");
        return(false);
      }
    }

    if (method == "Sum")
    {
      while (it != eit)
      {
        mesh->get_edges(edgearray, *(it));
        int nsize = edgearray.size();
        typename FIELD::value_type val;
        zero(val);
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
      typename FIELD::value_type test;
      if (has_compare(test))
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
      else
      {
        reporter->error("Median has not been implemented for this type of data");
        return(false);
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
        typename FIELD::value_type val;
        zero(val);
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
      typename FIELD::value_type test;
      if (has_compare(test))
      {
        while (it != eit)
        {
          mesh->get_faces(facearray, *(it));
          int nsize = facearray.size();
          typename FIELD::value_type val;
          zero(val);
          typename FIELD::value_type tval;
          zero(tval);
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
      else
      {
        reporter->error("Maximum has not been implemented for this type of data");
        return(false);
      }
    }
    
    if (method == "Min")
    {
      typename FIELD::value_type test;
      if (has_compare(test))
      {
        while (it != eit)
        {
          mesh->get_faces(facearray, *it);
          int nsize = facearray.size();
          typename FIELD::value_type val;
          zero(val);
          typename FIELD::value_type tval;
          zero(tval);
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
      else
      {
        reporter->error("Maximum has not been implemented for this type of data");
        return(false);
      }
    }

    if (method == "Sum")
    {
      while (it != eit)
      {
        mesh->get_faces(facearray, *(it));
        int nsize = facearray.size();
        typename FIELD::value_type val;
        zero(val);
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
      typename FIELD::value_type test;
      if (has_compare(test))
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
      else
      {
        reporter->error("Median has not been implemented for this type of data");
        return(false);
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
        typename FIELD::value_type val;
        zero(val);
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
      typename FIELD::value_type test;
      if (has_compare(test))
      {
        while (it != eit)
        {
          mesh->get_cells(cellarray, *(it));
          int nsize = cellarray.size();
          typename FIELD::value_type val;
          zero(val);
          typename FIELD::value_type tval;
          zero(tval);
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
      else
      {
        reporter->error("Maximum has not been implemented for this type of data");
        return(false);
      }
    }
    
    if (method == "Min")
    {
      typename FIELD::value_type test;
      if (has_compare(test))
      {
        while (it != eit)
        {
          mesh->get_cells(cellarray, *it);
          int nsize = cellarray.size();
          typename FIELD::value_type val;
          zero(val);
          typename FIELD::value_type tval;
          zero(tval);
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
      else
      {
        reporter->error("Maximum has not been implemented for this type of data");
        return(false);
      }
    }

    if (method == "Sum")
    {
      while (it != eit)
      {
        mesh->get_cells(cellarray, *(it));
        int nsize = cellarray.size();
        typename FIELD::value_type val;
        zero(val);
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
      typename FIELD::value_type test;
      if (has_compare(test))
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
      else
      {
        reporter->error("Median has not been implemented for this type of data");
        return(false);
      }      
    }
  }

  output = dynamic_cast<SCIRun::Field *>(ofield);
  return(true);
}


} // namespace ModelCreation

#endif
