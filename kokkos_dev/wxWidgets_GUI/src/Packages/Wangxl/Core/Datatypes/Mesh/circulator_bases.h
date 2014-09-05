// ======================================================================
//
// Copyright (c) 1997 The CGAL Consortium

// This software and related documentation are part of the Computational
// Geometry Algorithms Library (CGAL).
// This software and documentation are provided "as-is" and without warranty
// of any kind. In no event shall the CGAL Consortium be liable for any
// damage of any kind. 
//
// Every use of CGAL requires a license. 
//
// Academic research and teaching license
// - For academic research and teaching purposes, permission to use and copy
//   the software and its documentation is hereby granted free of charge,
//   provided that it is not a component of a commercial product, and this
//   notice appears in all copies of the software and related documentation. 
//
// Commercial licenses
// - A commercial license is available through Algorithmic Solutions, who also
//   markets LEDA (http://www.algorithmic-solutions.com). 
// - Commercial users may apply for an evaluation license by writing to
//   (Andreas.Fabri@geometryfactory.com). 
//
// The CGAL Consortium consists of Utrecht University (The Netherlands),
// ETH Zurich (Switzerland), Freie Universitaet Berlin (Germany),
// INRIA Sophia-Antipolis (France), Martin-Luther-University Halle-Wittenberg
// (Germany), Max-Planck-Institute Saarbrucken (Germany), RISC Linz (Austria),
// and Tel-Aviv University (Israel).
//
// ----------------------------------------------------------------------
//
// release       : CGAL-2.3
// release_date  : 2001, August 13
//
// file          : include/CGAL/circulator_bases.h
// package       : Circulator (3.17)
// chapter       : $CGAL_Chapter: Circulators $
// source        : circulator.fw
// revision      : $Revision$
// revision_date : $Date$
// author(s)     : Lutz Kettner
//
// coordinator   : INRIA, Sophia Antipolis
//
// Base classes and tags to build own circulators.
// email         : contact@cgal.org
// www           : http://www.cgal.org
//
// ======================================================================

#ifndef CGAL_CIRCULATOR_BASES_H
#define CGAL_CIRCULATOR_BASES_H 1

#ifndef CGAL_PROTECT_CSTDDEF
#include <cstddef>
#define CGAL_PROTECT_CSTDDEF
#endif
#ifndef CGAL_PROTECT_ITERATOR
#include <iterator>
#define CGAL_PROTECT_ITERATOR
#endif

CGAL_BEGIN_NAMESPACE

struct Circulator_tag {};                   // any circulator.
struct Iterator_tag {};                     // any iterator.

struct Forward_circulator_tag
    : public CGAL_STD::forward_iterator_tag {};
struct Bidirectional_circulator_tag
    : public CGAL_STD::bidirectional_iterator_tag {};
struct Random_access_circulator_tag
    : public CGAL_STD::random_access_iterator_tag {};
template <class T, class Dist = std::ptrdiff_t, class Size = std::size_t>
struct Forward_circulator_base {
    typedef T                            value_type;
    typedef Dist                         difference_type;
    typedef Size                         size_type;
    typedef T*                           pointer;
    typedef T&                           reference;
    typedef Forward_circulator_tag       iterator_category;
};
template <class T, class Dist = std::ptrdiff_t, class Size = std::size_t>
struct Bidirectional_circulator_base {
    typedef T                            value_type;
    typedef Dist                         difference_type;
    typedef Size                         size_type;
    typedef T*                           pointer;
    typedef T&                           reference;
    typedef Bidirectional_circulator_tag iterator_category;
};
template <class T, class Dist = std::ptrdiff_t, class Size = std::size_t>
struct Random_access_circulator_base {
    typedef T                            value_type;
    typedef Dist                         difference_type;
    typedef Size                         size_type;
    typedef T*                           pointer;
    typedef T&                           reference;
    typedef Random_access_circulator_tag iterator_category;
};
template < class Category,
           class T,
           class Distance  = std::ptrdiff_t,
           class Size      = std::size_t,
           class Pointer   = T*,
           class Reference = T&>
struct Circulator_base {
    typedef Category  iterator_category;
    typedef T         value_type;
    typedef Distance  difference_type;
    typedef Size      size_type;
    typedef Pointer   pointer;
    typedef Reference reference;
};

// variant base classes
// ---------------------
template <class T, class Dist = std::ptrdiff_t, class Size = std::size_t>
class Forward_circulator_ptrbase         // forward circulator.
{
    protected:
        void* _ptr;
    public:
        typedef Forward_circulator_tag  iterator_category;
        typedef T                           value_type;
        typedef Dist                        difference_type;
        typedef Size                        size_type;
        typedef T*                          pointer;
        typedef T&                          reference;
        Forward_circulator_ptrbase()        : _ptr(NULL) {}
        Forward_circulator_ptrbase(void* p) : _ptr(p) {}
};
template <class T, class Dist = std::ptrdiff_t, class Size = std::size_t>
class Bidirectional_circulator_ptrbase   // bidirectional circulator.
{
    protected:
        void* _ptr;
    public:
        typedef Bidirectional_circulator_tag  iterator_category;
        typedef T                           value_type;
        typedef Dist                        difference_type;
        typedef Size                        size_type;
        typedef T*                          pointer;
        typedef T&                          reference;
        Bidirectional_circulator_ptrbase()        : _ptr(NULL) {}
        Bidirectional_circulator_ptrbase(void* p) : _ptr(p) {}
};
template <class T, class Dist = std::ptrdiff_t, class Size = std::size_t>
class Random_access_circulator_ptrbase   // random access circulator.
{
    protected:
        void* _ptr;
    public:
        typedef Random_access_circulator_tag iterator_category;
        typedef T                           value_type;
        typedef Dist                        difference_type;
        typedef Size                        size_type;
        typedef T*                          pointer;
        typedef T&                          reference;
        Random_access_circulator_ptrbase()        : _ptr(NULL) {}
        Random_access_circulator_ptrbase(void* p) : _ptr(p) {}
};

CGAL_END_NAMESPACE

#endif // CGAL_CIRCULATOR_BASES_H //
// EOF //
