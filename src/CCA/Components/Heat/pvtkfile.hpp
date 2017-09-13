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

#ifndef Packages_Uintah_CCA_Components_Heat_pvtkfile_hpp
#define Packages_Uintah_CCA_Components_Heat_pvtkfile_hpp

#include <string>
#include <fstream>

#include <boost/property_tree/xml_parser.hpp>

namespace Uintah
{

class PVtkFile
{
    const std::string _path;
    const std::string _name;
    boost::property_tree::ptree _tree;

    PVtkFile ( const PVtkFile & other ) = delete;
    PVtkFile & operator= ( const PVtkFile & other ) = delete;
    bool operator== ( const PVtkFile & other ) const = delete;

public:
    inline PVtkFile ( const std::string & path, const std::string & name, double time );
    ~PVtkFile() = default;

    inline void add ( const std::string & filename );
    inline void add_time ( double time );
    inline void save ();
    inline std::string file_name ();
};

}

Uintah::PVtkFile::PVtkFile ( const std::string & path, const std::string & name, double time )
    : _path ( path ),
      _name ( name + ".pvti" )
{
    std::ifstream in ( _path + "/" + _name );
    bool exists = in.good();
    in.close();
    if ( exists )
    {
        boost::property_tree::read_xml ( _path + "/" + _name, _tree, boost::property_tree::xml_parser::trim_whitespace );
    }
    else
    {
        _tree.add ( "VTKFile.<xmlattr>.type","PImageData" );
        _tree.add ( "VTKFile.<xmlattr>.version","0.1" );
        add_time ( time );
    }
}

void Uintah::PVtkFile::add ( const std::string & filename )
{
    boost::property_tree::ptree child;
    child.add ( "<xmlattr>.Source",filename );
    _tree.add_child ( "VTKFile.PImageData.Piece", child );
}

void Uintah::PVtkFile::add_time(double time)
{
    boost::property_tree::ptree child;
    child.add ( "DataArray", time );
    child.add ( "DataArray.<xmlattr>.type","Float64" );
    child.add ( "DataArray.<xmlattr>.Name","TIME" );
    child.add ( "DataArray.<xmlattr>.NumberOfTuples","1" );
    child.add ( "DataArray.<xmlattr>.format","ascii");
    _tree.add_child ( "VTKFile.PImageData.FieldData", child );
}

void Uintah::PVtkFile::save ()
{
    boost::property_tree::xml_writer_settings<std::string> settings ( ' ', 4 );
    boost::property_tree::write_xml ( _path + "/" + _name, _tree, std::locale(), settings );
}

std::string Uintah::PVtkFile::file_name()
{
    return _name;
}

#endif // Packages_Uintah_CCA_Components_Heat_pvtkfile_hpp
