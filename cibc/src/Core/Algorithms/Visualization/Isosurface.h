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


/*
 *  Isosurface.cc:  
 *
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *
 *   Feb 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Matrix.h>


#include <Core/Algorithms/Visualization/share.h>

namespace SCIRun {

class SCISHARE IsosurfaceAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(vector<FieldHandle>& fields) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *iftd);
};


template< class IFIELD >
class IsosurfaceAlgoT : public IsosurfaceAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(vector<FieldHandle>& fields);
};


template< class IFIELD >
FieldHandle
IsosurfaceAlgoT<IFIELD>::execute(vector<FieldHandle>& fields)
{
  vector<IFIELD *> qfields(fields.size());
  for (unsigned int i=0; i < fields.size(); i++)
    qfields[i] = (IFIELD *)(fields[i].get_rep());
  
  return append_fields(qfields);
}

} // End namespace SCIRun


