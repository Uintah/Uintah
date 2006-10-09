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

/*
 *  ClipByFunction.cc:  Clip out parts of a field.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */
#include <Core/Datatypes/Field.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/HashTable.h>
#include <Core/Algorithms/Fields/ClipByFunction.h>

namespace SCIRun {

CompileInfoHandle
ClipByFunctionAlgo::get_compile_info(const TypeDescription *fsrc,
				     string clipFunction,
				     int hashoffset)
{
  unsigned int hashval = Hash(clipFunction, 0x7fffffff) + hashoffset;

  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_name("ClipByFunctionInstance" + to_string(hashval));
  static const string base_class_name("ClipByFunctionAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_name + "." +
		       fsrc->get_filename() + ".",
                       base_class_name, 
                       template_name, 
                       fsrc->get_name());

  // Add in the include path to compile this obj
  string class_declaration =
    string("template <class FIELD>\n") +
    "class " + template_name + " : public ClipByFunctionAlgoT<FIELD>\n" +
    "{\n" +
    "  using ClipByFunctionAlgoT<FIELD>::u0;\n" +
    "  using ClipByFunctionAlgoT<FIELD>::u1;\n" +
    "  using ClipByFunctionAlgoT<FIELD>::u2;\n" +
    "  using ClipByFunctionAlgoT<FIELD>::u3;\n" +
    "  using ClipByFunctionAlgoT<FIELD>::u4;\n" +
    "  using ClipByFunctionAlgoT<FIELD>::u5;\n" +
    "\n" +
    "  virtual bool vinside_p(double x, double y, double z,\n" +
    "                         typename FIELD::value_type v)\n" +
    "  {\n" +
    "    return " + clipFunction + ";\n" +
    "  }\n" +
    "\n" +
    "  virtual string identify()\n" +
    "  { return string(\"" + string_Cify(clipFunction) + "\"); }\n" +
    "};\n";

  rval->add_include(include_path);
  rval->add_post_include(class_declaration);
  fsrc->fill_compile_info(rval);

  return rval;
}

} // End namespace SCIRun

