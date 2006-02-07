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

#include <Packages/ModelCreation/Core/Fields/FieldDataElemToNode.h>

namespace ModelCreation {

using namespace SCIRun;

CompileInfoHandle
FieldDataElemToNodeAlgo::get_compile_info(FieldHandle field)
{
  const SCIRun::TypeDescription *basis_type = field->get_type_description(Field::BASIS_TD_E);
  const SCIRun::TypeDescription::td_vec *basis_subtype = basis_type->get_sub_type();
  const SCIRun::TypeDescription *data_type = (*basis_subtype)[0];
  const SCIRun::TypeDescription *meshtype = field->get_type_description(Field::MESH_TD_E);

  std::string mesh = meshtype->get_name();
  std::string basis = "";
  
  if (mesh.find("Scanline") != std::string::npos) basis = "CrvLinearLgn";
  if (mesh.find("Image") != std::string::npos)    basis = "QuadBilinearLgn";
  if (mesh.find("LatVol") != std::string::npos)   basis = "HexTrilinearLgn";
  if (mesh.find("Curve") != std::string::npos)    basis = "CrvLinearLgn";
  if (mesh.find("TriSurf") != std::string::npos)  basis = "TriLinearLgn";
  if (mesh.find("QuadSurf") != std::string::npos) basis = "QuadBilinearLgn";
  if (mesh.find("TetVol") != std::string::npos)   basis = "TetLinearLgn";
  if (mesh.find("PrismVol") != std::string::npos) basis = "PrismLinearLgn";
  if (mesh.find("HexVol") != std::string::npos)   basis = "HexTrilinearLgn";

  std::string datatype = data_type->get_name();
  basis = basis + "<" + datatype + " >";

  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  std::string include_path(SCIRun::TypeDescription::cc_to_h(__FILE__));
  std::string algo_name("FieldDataElemToNodeAlgoT");
  std::string base_name("FieldDataElemToNodeAlgo");

  std::string fieldtype_in = field->get_type_description()->get_name();
  std::string fieldtype_out = field->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() + "<" +
              field->get_type_description(Field::MESH_TD_E)->get_name() + "," + basis + "," +
              field->get_type_description(Field::FDATA_TD_E)->get_similar_name(datatype, 0,"<", "> ") + " > ";

  CompileInfoHandle ci = 
    scinew CompileInfo(algo_name + "." +
                       to_filename(fieldtype_in) + "." +    
                       to_filename(fieldtype_out) + ".",
                       base_name,
                       algo_name,  
                       fieldtype_in + "," + fieldtype_out);


  // Add in the include path to compile this obj
  ci->add_data_include(include_path);
  ci->add_namespace("ModelCreation");  
  ci->add_namespace("SCIRun");
  
  field->get_type_description()->fill_compile_info(ci.get_rep());
  return(ci);
}

} // namespace ModelCreation