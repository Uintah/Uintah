/*
 *  PackagePathIterator.cc: Iterates through the package paths
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

#include <stdlib.h>
#include <string.h>
#include "PackagePathIterator.h"

#ifndef DEFAULT_PACKAGE_PATH
#error You must set a DEFAULT_PACKAGE_PATH or life is pretty dull
#endif   

namespace Morgan {
    PackagePathIterator::PackagePathIterator(const char* separators) : 
        package_path_(0), curr_path_(0), separators_(separators) {
        package_path_ = getenv("PACKAGE_PATH");
        if(!package_path_) {
            package_path_ = DEFAULT_PACKAGE_PATH;
        }
        package_path_ = strdup(package_path_); // we need a private copy of this
        if(package_path_) {
            // start tokenizing
            curr_path_ = strtok(package_path_, separators); 
        }
    }

    PackagePathIterator::~PackagePathIterator() {
        free(package_path_);
    }

    const char* PackagePathIterator::operator * () const {
        return curr_path_;
    }

    PackagePathIterator& PackagePathIterator::operator ++ () {
        curr_path_ = strtok(NULL, separators_); 
        return *this;
    }

    PackagePathIterator::operator bool () const {
        return curr_path_ != NULL;
    }
}
