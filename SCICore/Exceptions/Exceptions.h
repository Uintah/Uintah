
/*
 *  Assert.h: Utility for providing a transition to a real exception model
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Containers_Exceptions_h
#define SCI_Containers_Exceptions_h 1

#include <share/share.h>

// Do this until we get throw to work...
#define EXCEPTION(exc) \
	Exception(exc, __FILE__, __LINE__)

namespace SCICore {
namespace ExceptionsSpace {

class SHARE AllExceptions {
public:
    virtual ~AllExceptions();
    virtual void print_error() const=0;
};

#ifdef _WIN32
SHARE void Exception(const AllExceptions&, char*, int);
#else
extern void Exception(const AllExceptions&, char*, int);
#endif

class SHARE General : public AllExceptions {
    char* name;
public:
    General(char* name);
    virtual ~General();
    virtual void print_error() const;
};

class SHARE AssertionFailed : public AllExceptions {
    char* condition;
public:
    AssertionFailed(char* condition);
    virtual ~AssertionFailed();
    virtual void print_error() const;
};

class SHARE AssertionEQFailed : public AllExceptions {
    char* c1;
    char* c2;
    int v1;
    int v2;
public:
    AssertionEQFailed(char* c1, char* c2, int v1, int v2);
    virtual ~AssertionEQFailed();
    virtual void print_error() const;
};

class SHARE AssertionRangeFailed : public AllExceptions {
    char* exp_text;
    char* lower_text;
    char* upper_text;
    int exp;
    int lower;
    int upper;
public:
    AssertionRangeFailed(char* exp_text, char* lower_text,
			 char* upper_text, int exp, int lower, int upper);
    virtual ~AssertionRangeFailed();
    virtual void print_error() const;
};

class SHARE ErrorOpeningFile : public AllExceptions {
    char* name;
public:
    ErrorOpeningFile(char* name);
    virtual ~ErrorOpeningFile();
    virtual void print_error() const;
};

class SHARE UnInitialized : public AllExceptions {
    char* name;
public:
    UnInitialized(char* name);
    virtual ~UnInitialized();
    virtual void print_error() const;
};

} // End namespace ExceptionsSpace
} // End namespace SCICore

//
// $log$
//

#endif
