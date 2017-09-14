/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef Packages_Uintah_CCA_Components_Heat_vtkfile_hpp
#define Packages_Uintah_CCA_Components_Heat_vtkfile_hpp

#include <iomanip>
#include <fstream>
#include <sstream>

#include <vtkSmartPointer.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>

#define ID_WIDTH 5

class vtkImageData;

namespace Uintah
{

class VtkFile
{
    const std::string _path;
    std::string _name;
    vtkSmartPointer<vtkImageData> _grid;

    VtkFile ( const VtkFile & other ) = delete;
    VtkFile & operator= ( const VtkFile & other ) = delete;
    bool operator== ( const VtkFile & other ) const = delete;

public:
    inline VtkFile ( const std::string & path, const unsigned & id );
    ~VtkFile();

    inline bool exists();

    inline std::string file_name();
//     inline std::string rel_path();

    void set_grid ( const double & hx, const double & hy, const double & hz, const int & Nx, const int & Ny, const int & Nz, const double & x0, const double & y0, const double & z0 );
    void add_time ( const double & time );
    void add_node_data ( const std::string & name, const constNCVariable<double> & var, const IntVector & low, const IntVector & high );
    void add_cell_data ( const std::string & name, const constCCVariable<int> & var, const IntVector & low, const IntVector & high );
    void add_cell_data ( const std::string & name, const constCCVariable<double> & var, const IntVector & low, const IntVector & high );
    void save();
};

}

Uintah::VtkFile::VtkFile ( const std::string & path, const unsigned & id )
    : _path ( path ),
      _grid ( )
{
    std::stringstream stream;
    stream << "p" << std::setw ( ID_WIDTH ) << std::setfill ( '0' ) << id << ".vti";
    _name = stream.str();
}

bool Uintah::VtkFile::exists()
{
//     std::string filepath = ;
    std::ifstream file ( _path + "/" + _name );
    return file.good();
}

std::string Uintah::VtkFile::file_name()
{
    return _name;
}

// std::string Uintah::VtkFile::abs_path()
// {
//     return _path + "/" + _data + "/" + _name;
// }
// 
// std::string Uintah::VtkFile::rel_path()
// {
//     return _data + "/" + _name;
// }

#endif // Packages_Uintah_CCA_Components_Heat_vtkfile_hpp

