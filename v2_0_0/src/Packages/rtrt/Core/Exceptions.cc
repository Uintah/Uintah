
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

#ifdef __GNUG__
#pragma implementation
#endif

#include <Packages/rtrt/Core/Exceptions.h>
#include <iostream>
using namespace std;
#include <stdlib.h>		// for abort()
#include <string.h>

// The exception subroutine...
// Print an error and abort
void Exception(const AllExceptions& exc, char* file, int line)
{
	exc.print_error();
	cerr << "At line " << line << " in " << file << endl;
	abort();
}

General::General(char* name)
: name(strdup(name))
{
}


void General::print_error() const
{
	cerr << "Exception: " << name << endl;
}

AssertionFailed::AssertionFailed(char* condition)
: condition(strdup(condition))
{
}

void AssertionFailed::print_error() const
{
	cerr << "Assertion Failed: " << condition << endl;
}

AssertionEQFailed::AssertionEQFailed(char* c1, char* c2, int v1, int v2)
: c1(strdup(c1)), c2(strdup(c2)), v1(v1), v2(v2)
{
}

void AssertionEQFailed::print_error() const
{
    cerr << "Assertion failed: "
	 << c1 << "(" << v1 << ") != " 
	 << c2 << "(" << v2 << ")" << endl;
}

AssertionRangeFailed::AssertionRangeFailed(char* exp_text, char* lower_text,
					   char* upper_text, int exp,
					   int lower, int upper)
: exp_text(strdup(exp_text)), lower_text(strdup(lower_text)),
  upper_text(strdup(upper_text)), exp(exp), lower(lower), upper(upper)
{
}

void AssertionRangeFailed::print_error() const
{
	cerr << "Assertion Failed: "
	     << lower_text << "(" << lower << ") <= "
	     << exp_text << "(" << exp << ") < "
	     << upper_text << "(" << upper << ")" << endl;
}

ErrorOpeningFile::ErrorOpeningFile(char* name)
: name(strdup(name))
{
}

void ErrorOpeningFile::print_error() const
{
	cerr << "Error Opening File: " << name << endl;
}

UnInitialized::UnInitialized(char* name)
: name(strdup(name))
{
}

void UnInitialized::print_error() const
{
	cerr << "Uninitialized " << name << endl;
}

AllExceptions::~AllExceptions()
{
}

General::~General()
{
}

AssertionFailed::~AssertionFailed()
{
}

AssertionEQFailed::~AssertionEQFailed()
{
}

AssertionRangeFailed::~AssertionRangeFailed()
{
}

ErrorOpeningFile::~ErrorOpeningFile()
{
}

UnInitialized::~UnInitialized()
{
}
