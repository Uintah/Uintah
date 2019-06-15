/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

/**
 * @file CCA/Components/PhaseField/Exceptions/IntVectorOutOfBounds.cc
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_Exceptions_IntVectorOutOfBounds_h
#define Packages_Uintah_CCA_Components_PhaseField_Exceptions_IntVectorOutOfBounds_h

#include <Core/Exceptions/Exception.h>
#include <Core/Geometry/IntVector.h>

namespace Uintah
{
namespace PhaseField
{

/// IntVectorOutOfBounds Exception
class IntVectorOutOfBounds
    : public Exception
{
private: // MEMBERS

    /// out of bound IntVector
    IntVector m_value;

    /// error message
    char * m_msg;

public: // CONSTRUCTORS/DESTRUCTOR

    /// copy assignment
    IntVectorOutOfBounds & operator= ( const IntVectorOutOfBounds );

    /// copy constructor
    IntVectorOutOfBounds (
        const IntVectorOutOfBounds &
    );

    /**
     * @brief Constuctor
     *
     * @param value out of bound IntVector
     * @param file file name from which the exception is thrown
     * @param line line number from which the exception is thrown
     */
    IntVectorOutOfBounds (
        const IntVector & value,
        const char * file,
        int line
    );

    /// default destructor
    virtual ~IntVectorOutOfBounds() = default;

public: // METHODS

    /// get error message
    virtual const char * message() const;

    /// get exception name
    virtual const char * type() const;

}; // class IntVectorOutOfBounds

} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_Exceptions_IntVectorOutOfBounds_h
