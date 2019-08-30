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

#include <CCA/Components/PhaseField/Exceptions/IntVectorOutOfBounds.h>
#include <CCA/Components/PhaseField/Util/Definitions.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

using namespace Uintah;
using namespace PhaseField;

IntVectorOutOfBounds::IntVectorOutOfBounds (
    const IntVector & value,
    const char * file,
    int line
) : m_value ( value )
{
    std::ostringstream s;
    s << "An IntVectorOutOfBounds exception was thrown.\n"
      << file << ":" << line << "\n"
      << "index " << value;
    m_msg = strdup ( s.str().c_str() );
#ifdef EXCEPTIONS_CRASH
    std::cout << m_msg << std::endl;
#endif
}

IntVectorOutOfBounds::IntVectorOutOfBounds (
    const IntVectorOutOfBounds & copy
)
    : m_value ( copy.m_value ),
      m_msg ( strdup ( copy.m_msg ) )
{
}

const char * IntVectorOutOfBounds::message() const
{
    return m_msg;
}

const char * IntVectorOutOfBounds::type() const
{
    return "IntVectorOutOfBounds";
}
