//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
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
