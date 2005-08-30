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


//    File   : ChangeFieldDataType.h
//    Author : McKay Davis
//    Date   : July 2002

#if !defined(ChangeFieldDataType_h)
#define ChangeFieldDataType_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Field.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>

namespace SCIRun {


class ChangeFieldDataTypeAlgoCreate : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(FieldHandle fsrc_h) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const string &fdstname);
};


template <class FSRC, class FOUT>
class ChangeFieldDataTypeAlgoCreateT : public ChangeFieldDataTypeAlgoCreate
{
public:

  virtual FieldHandle execute(FieldHandle fsrc_h);
};


template <class FSRC, class FOUT>
FieldHandle
ChangeFieldDataTypeAlgoCreateT<FSRC, FOUT>::execute(FieldHandle fsrc_h)
{
  FSRC *fsrc = dynamic_cast<FSRC *>(fsrc_h.get_rep());

  // Create the field with the new mesh and data location.
  FOUT *fout = scinew FOUT(fsrc->get_typed_mesh(), fsrc_h->basis_order());

  // Copy the properties from old field to new field.
  fout->copy_properties(fsrc);

  return fout;
}

class ChangeFieldDataTypeAlgoCopy : public DynamicAlgoBase
{
public:

  virtual void execute(FieldHandle fsrc_h, FieldHandle fout_h) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *fdst);
};


template <class FSRC, class FOUT>
class ChangeFieldDataTypeAlgoCopyT : public ChangeFieldDataTypeAlgoCopy
{
public:

  //! virtual interface. 
  virtual void execute(FieldHandle fsrc_h, FieldHandle fout_h);
};


template <class FSRC, class FOUT>
void
ChangeFieldDataTypeAlgoCopyT<FSRC, FOUT>::execute(FieldHandle fsrc_h,
					FieldHandle fout_h)
{
  FSRC *fsrc = dynamic_cast<FSRC *>(fsrc_h.get_rep());
  FOUT *fout = dynamic_cast<FOUT *>(fout_h.get_rep());

  typename FSRC::fdata_type::iterator in = fsrc->fdata().begin();
  typename FOUT::fdata_type::iterator out = fout->fdata().begin();
  typename FSRC::fdata_type::iterator end = fsrc->fdata().end();
  if (fout->basis_order() == fsrc->basis_order())
  {
    while (in != end)
    {
      *out = (typename FOUT::value_type)(*in);
      ++in; ++out;
    }
  }
}


} // end namespace SCIRun

#endif // ChangeFieldDataType_h
