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

#ifndef Packages_Uintah_CCA_Components_Heat_TimeScheme_h
#define Packages_Uintah_CCA_Components_Heat_TimeScheme_h

namespace Uintah
{

enum TimeScheme : unsigned
{
    Unknown       = 0x00000000,

    Explicit      = 0x01000000,
    ForwardEuler  = 0x01000001,

    Implicit      = 0x02000000,
    BackwardEuler = 0x02000001,
    CrankNicolson = 0x02000002

}; // enum TimeScheme

inline TimeScheme from_str ( const std::string & value );

} // namespace Uintah

inline Uintah::TimeScheme Uintah::from_str ( const std::string & value )
{
    if ( value == "forward_euler" ) return TimeScheme::ForwardEuler;
    if ( value == "backward_euler" ) return TimeScheme::BackwardEuler;
    if ( value == "crank_nicolson" ) return TimeScheme::CrankNicolson;
    return TimeScheme::Unknown;
}

#endif // Packages_Uintah_CCA_Components_Heat_TimeScheme_h
