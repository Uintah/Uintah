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

//    File   : ConvertTet.h
//    Author : Martin Cole
//    Date   : Tue Feb 26 22:11:00 2002


#if !defined(Quadratic_ConvertTet_h)
#define Quadratic_ConvertTet_h

#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/QuadraticTetVolField.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

namespace SCIRun {

//! ConvertTetBase supports the dynamically loadable algorithm concept.
//! when dynamically loaded the user will dynamically cast to a 
//! ConvertTetBase from the DynamicAlgoBase they will have a pointer to.
class ConvertTetBase : public DynamicAlgoBase
{
public:
  virtual FieldHandle convert_quadratic(FieldHandle in) = 0;
  virtual ~ConvertTetBase();

  static const string& get_h_file_path();
  static string dyn_file_name(const TypeDescription *td) {
    // add no extension.
    return template_class_name() + "." + td->get_filename() + ".";
  }

  static const string base_class_name() {
    static string name("ConvertTetBase");
    return name;
  }

  static const string template_class_name() {
    static string name("ConvertTet");
    return name;
  }

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *td);
};

template <class Fld>
class ConvertTet : public ConvertTetBase
{
public:
  //! virtual interface.
  virtual FieldHandle convert_quadratic(FieldHandle in);
};

template <class Fld>
FieldHandle
ConvertTet<Fld>::convert_quadratic(FieldHandle ifh)
{
  Fld *fld = dynamic_cast<Fld*>(ifh.get_rep());
  ASSERT(fld != 0);
  
  typedef typename Fld::value_type val_t;
  FieldHandle fh(QuadraticTetVolField<val_t>::create_from(*fld));
  return fh;
}

} // end namespace SCIRun

#endif // Quadratic_ConvertTet_h
