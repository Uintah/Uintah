/*
 *  PackagePathIterator.h: Iterates through the package paths
 *
 *  Written by:
 *   Jason V. Morgan
 *   Department of Computer Science
 *   University of Utah
 *   December 2000
 *
 *  Updated by:
 *   Jason V. Morgan
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef Morgan_PackagePathIterator_h
#define Morgan_PackagePathIterator_h

namespace Morgan {
    class PackagePathIterator {
    public:
        //
        // PackagePathIterator
        // in:  const char* separators
        //
        // Constructor for the iterator, starts it off with the first item.
        // It gathers this from either the PACKAGE_PATH environment variable
        // if it exists, or the DEFAULT_PACKAGE_PATH data if PACKAGE_PATH not
        // set.  A package path is a set of paths seperated by any of the the 
        // given separator characters (defaults to ".").
        //
        PackagePathIterator(const char* separators = ",");

        ~PackagePathIterator();

        //
        // const char* operator * const
        //
        // Returns the current package path string.  For example, given the
        // package path "/usr,/opt,/home", this function would return "/usr"
        // at first, "/opt" after the ++ operator was called, and "/home",
        // after the ++ operator was called again.  If no more strings are
        // available, this function returns NULL.
        //
        const char* operator * () const;

        //
        // PackagePathIterator& operator ++
        //
        // Goes to the next package path string available.  Note that only the
        // prefix notation is allowed.
        //
        PackagePathIterator& operator ++ ();

        //
        // operator bool const
        //
        // Returns true if any strings are left and false if otherwise.
        //
        operator bool () const;

    private:

        char* package_path_;
        const char* separators_;
        char* curr_path_;
    };
}

#endif // Morgan_PackagePathIterator_h
