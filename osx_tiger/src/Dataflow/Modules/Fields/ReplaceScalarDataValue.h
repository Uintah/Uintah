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


//    File   : ReplaceScalarDataValue.h
//    Author : David Weinstein
//    Date   : October 2002

#if !defined(ReplaceScalarDataValue_h)
#define ReplaceScalarDataValue_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

namespace SCIRun {

class ReplaceScalarDataValueAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src, double oldvalue, 
			      double newvalue) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *lsrc);
};


template <class FIELD, class LOC>
class ReplaceScalarDataValueAlgoT : public ReplaceScalarDataValueAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src, double oldvalue, 
			      double newvalue);
};


template <class FIELD, class LOC>
FieldHandle
ReplaceScalarDataValueAlgoT<FIELD, LOC>::execute(FieldHandle field_h,
					double oldvalue,
					double newvalue)
{
  FIELD *ifield = dynamic_cast<FIELD *>(field_h.get_rep());
  FIELD *ofield = ifield->clone();

  typename FIELD::fdata_type::iterator ibi, iei, obi;
  ibi = ifield->fdata().begin();
  iei = ifield->fdata().end();
  obi = ofield->fdata().begin();

  while (ibi != iei)
  {
    typename FIELD::value_type val = *ibi;
    double tmp = (double)val;
    if (tmp == oldvalue) tmp=newvalue;
    *obi = (typename FIELD::value_type)(tmp);
    ++ibi;
    ++obi;

  }

  return ofield;
}


} // end namespace SCIRun

#endif // ReplaceScalarDataValue_h
