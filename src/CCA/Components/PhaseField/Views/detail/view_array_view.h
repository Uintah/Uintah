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
 * @file CCA/Components/PhaseField/Views/detail/view_array_view.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_Views_detail_view_array_view_h
#define Packages_Uintah_CCA_Components_PhaseField_Views_detail_view_array_view_h

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

template <typename Field> class view;

/**
 * @brief Container for collecting multiple views as an array
 *
 * It is used as base class for view_array in order to provide a unique container
 * across all view_array derived classes implementations
 *
 * @remark no instantiation must be performed here
 *
 * @tparam Field type of Field
 * @tparam N size of the array
 */
template <typename Field, size_t N >
class view_array_view
{
protected: // MEMBERS

    std::array < view<Field> *, N > m_view_ptr;

protected: // CONSTRUCTOR

    /**
     * @brief Constructor
     *
     * Initialize the array without instantiating component views
     */
    view_array_view ()
    {
        m_view_ptr.fill ( nullptr );
    }

}; // class view_array_view

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_Views_detail_view_array_view_h
