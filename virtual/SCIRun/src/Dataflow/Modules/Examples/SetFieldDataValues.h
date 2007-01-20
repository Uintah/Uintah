//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : ChangeTetVolScalars.cc
//    Author : Martin Cole
//    Date   : Mon Sep 11 11:22:14 2006



#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/GenericField.h>

#include <iostream>

namespace SCIRun {
using namespace std;
using namespace SCIRun;



class SetFieldDataValuesAlgoBase : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(ProgressReporter *, FieldHandle,
			      double) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *td);
};

template <class Fld>
class SetFieldDataValuesT : public SetFieldDataValuesAlgoBase
{
public:
  //! virtual interface.
  virtual FieldHandle execute(ProgressReporter *, 
			      FieldHandle,
			      double);
};


template <class Fld>
FieldHandle
SetFieldDataValuesT<Fld>::execute(ProgressReporter *reporter, 
			     FieldHandle ifh, 
			     double newval)
{

  //Must detach since we will be altering the input field.
  ifh.detach();
  
  Fld* in = dynamic_cast<Fld*>(ifh.get_rep());
  if (! in) {
    cerr << "Input field type does not match algorithm paramter type." 
	 << endl;
    return 0;
  }

  typedef typename Fld::mesh_type Msh;

  typename Fld::mesh_handle_type mh = in->get_typed_mesh();
  typename Msh::Node::iterator iter;
  typename Msh::Node::iterator end;

  mh->synchronize(Mesh::NODES_E);

  mh->begin(iter);
  mh->end(end);
  while (iter != end) {
    typename Msh::Node::index_type ni = *iter;
    Point node;
    mh->get_center(node, ni);
    typename Fld::value_type val;
    in->value(val, ni);
    cerr << "at point: " << node << " the input value is: " << val << endl;

    // Set the value to be the value from the gui;
    in->set_value(newval, ni);
    ++iter;
  }

  return ifh;

}

} // End namespace SCIRun


