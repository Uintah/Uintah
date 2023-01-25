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


#ifndef Packages_Uintah_CCA_Components_PhaseField_Support_h
#define Packages_Uintah_CCA_Components_PhaseField_Support_h

#include <list>
#include <Core/Grid/Region.h>

namespace Uintah
{
namespace PhaseField
{

/**
 * @brief View Domain
 *
 * It wraps a list of rectangular regions
 */
class Support
    : public std::list<Region>
{
private: // MEMBERS

public: // CONSTRUCTOR

    using std::list<Region>::list;

private: // METHODS

    /**
     * @brief function for sorting
     *
     * @param region1 first member of the comparison
     * @param region2 second member of the comparison
     * @return if region1 precedes region2 in lexicografic order
     */
    static bool
    compare_regions (
        Region & region1,
        Region & region2
    )
    {
        if ( region1.low().x() < region2.low().x() ) return true;
        if ( region1.low().x() > region2.low().x() ) return false;
        if ( region1.low().y() < region2.low().y() ) return true;
        if ( region1.low().y() > region2.low().y() ) return false;
        if ( region1.low().z() < region2.low().z() ) return true;
        if ( region1.low().z() > region2.low().z() ) return false;
        if ( region1.high().x() < region2.high().x() ) return true;
        if ( region1.high().x() > region2.high().x() ) return false;
        if ( region1.high().y() < region2.high().y() ) return true;
        if ( region1.high().y() > region2.high().y() ) return false;
        if ( region1.high().z() < region2.high().z() ) return true;
        if ( region1.high().z() > region2.high().z() ) return false;
        return false;
    }

public: // METHODS

    /// Reduce the number of region in the list
    void simplify()
    {
        size_t n;
        do
        {
            n = size();

            iterator it1 = begin();
            while ( it1 != end() )
                if ( it1->degenerate() )
                    it1 = erase ( it1 );
                else
                    ++it1;

            sort ( compare_regions ); // order by x, then by y, then by z

            // join region with same low.x,y high.x,y

            it1 = begin();
            if ( it1 == end() ) return;

            iterator it2 = std::next ( it1 );

            while ( it2 != end () )
            {
                if ( it1->low().x() == it2->low().x() && it1->high().x() == it2->high().x() &&
                        it1->low().y() == it2->low().y() && it1->high().y() == it2->high().y() )
                {
                    if ( it1->low().z() <= it2->low().z() && it2->low().z() <= it1->high().z() )
                    {
                        it1->high().z ( it2->high().z() );
                        it2 = erase ( it2 );
                    }
                    else
                    {
                        it1 = it2;
                        ++it2;
                    }

                }
                else
                {
                    ++it1;
                    ++it2;
                }
            }
            //no more degenerate regions!

            // join region with same low.x,z high.x,z
            it1 = begin();
            it2 = std::next ( it1 );
            while ( it2 != end () )
            {
                if ( it1->low().x() == it2->low().x() && it1->high().x() == it2->high().x() &&
                        it1->low().z() == it2->low().z() && it1->high().z() == it2->high().z() )
                {
                    if ( it1->low().y() <= it2->low().y() && it2->low().y() <= it1->high().y() )
                    {
                        it1->high().y ( it2->high().y() );
                        it2 = erase ( it2 );
                    }
                    else
                    {
                        it1 = it2;
                        ++it2;
                    }
                }
                else
                {
                    ++it1;
                    ++it2;
                }
            }

            // join region with same low.y,z high.y,z
            it1 = begin();
            it2 = std::next ( it1 );
            while ( it2 != end () )
            {
                if ( it1->low().y() == it2->low().y() && it1->high().y() == it2->high().y() &&
                        it1->low().z() == it2->low().z() && it1->high().z() == it2->high().z() )
                {
                    if ( it1->low().x() <= it2->low().x() && it2->low().x() <= it1->high().x() )
                    {
                        it1->high().x ( it2->high().x() );
                        it2 = erase ( it2 );
                    }
                    else
                    {
                        it1 = it2;
                        ++it2;
                    }
                }
                else
                {
                    ++it1;
                    ++it2;
                }
            }
        }
        while ( size() && size() != n ); // repeat untile no simplification is made
    };
};

} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_Support_h
