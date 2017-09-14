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

#ifndef Packages_Uintah_CCA_Components_Heat_visitfile_hpp
#define Packages_Uintah_CCA_Components_Heat_visitfile_hpp

#include <string>
#include <fstream>
#include <set>

namespace Uintah
{

class VisitFile
{
    const std::string _path;
    const std::string _name;
    std::set<std::string> _set;

    VisitFile ( const VisitFile & other ) = delete;
    VisitFile & operator= ( const VisitFile & other ) = delete;
    bool operator== ( const VisitFile & other ) const = delete;

public:
    inline VisitFile ( const std::string & path, const std::string & name, const bool & append );
    ~VisitFile() = default;

    inline void add ( const std::string & filename );
};

}

Uintah::VisitFile::VisitFile ( const std::string & path, const std::string & name, const bool & append )
    : _path ( path ),
      _name ( name )
{
    std::ofstream out;
    out.open ( _path + "/" + _name + ".visit", append ? std::ios_base::app : std::ios_base::trunc );
    out.close();
}

void Uintah::VisitFile::add ( const std::string & filename )
{
    auto ret = _set.insert ( filename );
    if ( ret.second )
    {
        std::ofstream out;
        out.open ( _path + "/" + _name + ".visit", std::ios_base::app );
        out << filename << std::endl;
        out.close();
    }
}

#endif // Packages_Uintah_CCA_Components_Heat_visitfile_hpp
