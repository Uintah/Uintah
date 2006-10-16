/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

#ifndef CORE_ALGORITHMS_FIELDS_ISINSIDEFIELD_H
#define CORE_ALGORITHMS_FIELDS_ISINSIDEFIELD_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>

namespace SCIRunAlgo {

using namespace SCIRun;

class IsInsideFieldAlgo : public DynamicAlgoBase
{
public:
  virtual bool IsInsideField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle object, double newval,double defval, std::string output_type, std::string basis_type, bool partial_inside = false);
};

template<class FSRC, class FDST, class FOBJ>
class IsInsideFieldAlgoT : public IsInsideFieldAlgo
{
public:
  virtual bool IsInsideField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle object, double newval,double defval, std::string output_type, std::string basis_type, bool partial_inside);
};


template<class FSRC, class FDST, class FOBJ>
bool IsInsideFieldAlgoT<FSRC,FDST,FOBJ>::IsInsideField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle objectfield, double newval,double defval, std::string output_type, std::string basis_type, bool partial_inside)
{
  FOBJ* objfield = dynamic_cast<FOBJ* >(objectfield.get_rep());
  if (objfield == 0)
  {
    pr->error("IsInsideField: Object is not valid");
    return(false);
  }

  FSRC* ifield = dynamic_cast<FSRC* >(input.get_rep());
  if (ifield == 0)
  {
    pr->error("IsInsideField: There is no input field");
    return(false);
  }

  typename FSRC::mesh_type *imesh = ifield->get_typed_mesh().get_rep();
  typename FOBJ::mesh_type *objmesh = objfield->get_typed_mesh().get_rep();

  FDST* ofield;
  
  if (output.get_rep() == 0)
  {
    ofield = scinew FDST(imesh);
    if (ofield == 0)
    {
      pr->error("IsInsideField: Could not create output field");
      return(false);
    }

    ofield->resize_fdata();
    output = dynamic_cast<SCIRun::Field* >(ofield);  

    if (ofield->basis_order() == 0)
    {
      typename FSRC::mesh_type::Elem::iterator it, it_end;

      imesh->begin(it); 
      imesh->end(it_end);

      while (it != it_end)
      {
        ofield->set_value(static_cast<typename FDST::value_type>(defval),*(it));
        ++it;
      }
    }
    else
    {
      typename FSRC::mesh_type::Node::iterator it, it_end;

      imesh->begin(it); 
      imesh->end(it_end);

      while (it != it_end)
      {
        ofield->set_value(static_cast<typename FDST::value_type>(defval),*(it));
        ++it;
      }    
    }
  }
  else
  {
    ofield = dynamic_cast<FDST *>(output.get_rep());
  }
  
  objmesh->synchronize(Mesh::LOCATE_E);


  if (ofield->basis_order() == 0)
  {
    typename FSRC::mesh_type::Elem::iterator it, it_end;
    typename FDST::value_type val;
 
    imesh->begin(it); 
    imesh->end(it_end);

    while (it != it_end)
    {
      Point p;
      typename FSRC::mesh_type::Node::array_type nodes;
      typename FOBJ::mesh_type::Elem::index_type cidx;
          
      if (partial_inside) val = 0; else val = 1;
      imesh->get_nodes(nodes,*it);
      for (int r=0; r< nodes.size(); r++)
      {
        imesh->get_center(p,nodes[r]);
        if (objmesh->locate(cidx,p))
        {
          if (partial_inside) val = 1;
          else val *= 1; // it is inside
        }
        else
        {
          if (!partial_inside) val *= 0;
        }
      }

      if (val == 1) ofield->set_value(static_cast<typename FDST::value_type>(newval),*(it));
      ++it;
    }
  }
  else if (ofield->basis_order() == 1)
  {
    typename FSRC::mesh_type::Node::iterator it, it_end;
    typename FDST::value_type val;
    typename FOBJ::mesh_type::Elem::index_type cidx;
     
    imesh->begin(it); 
    imesh->end(it_end);

    while (it != it_end)
    {
      Point p;
      imesh->get_center(p,*it);
      if (objmesh->locate(cidx,p))
      {
        val = 1; // it is inside    
      }
      else
      {
        val = 0;       
      }
      
      if (val == 1) ofield->set_value(static_cast<typename FDST::value_type>(newval),*(it));
      ++it;
    }
  }
  else
  {
    pr->error("IsInsideField: Cannot add distance data to field");
    return (false);  
  }
  return (true);
}

} // end namespace SCIRunAlgo

#endif
