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

#ifndef MODELCREATION_CORE_UTIL_FIELDINFORMATION
#define MODELCREATION_CORE_UTIL_FIELDINFORMATION 1

#include <Core/Util/TypeDescription.h>
#include <Core/Datatypes/Field.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>>

namespace SCIRun {

class FieldInformation {
  
  public:
    FieldInformation(FieldHandle handle);
  
    std::string get_field_type();
    void        set_field_type(std::string);

    std::string get_mesh_type();
    void        set_mesh_type(std::string);
    
    std::string get_mesh_basis_type();
    void        set_mesh_basis_type(std::string);

    std::string get_point_type();
    void        set_point_type(std::string);

    std::string get_basis_type();
    void        set_basis_type(std::string);

    std::string get_data_type();
    void        set_data_type(std::string);

    std::string get_container_type();
    void        set_container_type(std::string);

    std::string get_field_name();
    std::string get_field_filename();
    
    void fill_compile_info(CompileInfoHandle &ci);
  
    bool        is_isomorphic();
    bool        is_nonlinear();
    bool        is_linear();
    
    bool        is_nodata();
    bool        is_constantdata();
    bool        is_lineardata();
    bool        is_nonlineardata();
    bool        is_quadraticdata();
    bool        is_cubichmtdata();
    
    bool        is_constantmesh();
    bool        is_linearmesh();
    bool        is_nonlinearmesh();
    bool        is_quadraticmesh();
    bool        is_cubichmtmesh();
  
    bool        is_tensor();
    bool        is_vector();
    bool        is_scalar();
    bool        is_double();
    bool        is_float();
    bool        is_dvt();
    
    bool        is_regularmesh();
    bool        is_structuredmesh();
    bool        is_unstructuredmesh();
    
    bool        is_pointcloud();
    bool        is_curve();
    bool        is_surface();
    bool        is_volume();
    
    bool        make_nodata();
    bool        make_constantdata();
    bool        make_lineardata();
    bool        make_quadraticdata();
    bool        make_cubichmtdata();    
    
  private:
    // type names
    std::string field_type;
    std::string mesh_type;
    std::string mesh_basis_type;
    std::string point_type;
    std::string basis_type;
    std::string data_type;
    std::string container_type;
    
    // include files
    std::string field_type_h;
    std::string mesh_type_h;
    std::string mesh_basis_type_h;
    std::string point_type_h;
    std::string basis_type_h;
    std::string data_type_h;
    std::string container_type_h;

};

} // end namespace

#endif

