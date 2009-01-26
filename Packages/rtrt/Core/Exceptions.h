/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



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

#ifndef SCI_Classlib_Exceptions_h
#define SCI_Classlib_Exceptions_h 1

// Do this until we get throw to work...
#define EXCEPTION(exc) \
	::Exception(exc, __FILE__, __LINE__)

#ifdef __GNUG__
#pragma interface
#endif

//namespace rtrt {

class AllExceptions {
public:
    virtual ~AllExceptions();
    virtual void print_error() const=0;
};

extern void Exception(const AllExceptions&, char*, int);

class General : public AllExceptions {
    char* name;
public:
    General(char* name);
    virtual ~General();
    virtual void print_error() const;
};

class AssertionFailed : public AllExceptions {
    char* condition;
public:
    AssertionFailed(char* condition);
    virtual ~AssertionFailed();
    virtual void print_error() const;
};

class AssertionEQFailed : public AllExceptions {
    char* c1;
    char* c2;
    int v1;
    int v2;
public:
    AssertionEQFailed(char* c1, char* c2, int v1, int v2);
    virtual ~AssertionEQFailed();
    virtual void print_error() const;
};

class AssertionRangeFailed : public AllExceptions {
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

class ErrorOpeningFile : public AllExceptions {
    char* name;
public:
    ErrorOpeningFile(char* name);
    virtual ~ErrorOpeningFile();
    virtual void print_error() const;
};

class UnInitialized : public AllExceptions {
    char* name;
public:
    UnInitialized(char* name);
    virtual ~UnInitialized();
    virtual void print_error() const;
};

//} // end namespace rtrt

#endif
