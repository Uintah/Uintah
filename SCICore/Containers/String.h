
/*
 *  String.h: interface to String utility class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_SCICore_String_h
#define SCI_SCICore_String_h 1

#include <SCICore/share/share.h>

#include <iosfwd>

#ifdef DEFINE_OLD_IOSTREAM_OPERATORS
class istream;
class ostream;
#endif

namespace SCICore {
  namespace PersistentSpace {
    class Piostream;
  }
  namespace Tester {
    class RigorousTest;
    class PerfTest;
  }

namespace Containers {

using SCICore::Tester::RigorousTest;
using SCICore::Tester::PerfTest;
using SCICore::PersistentSpace::Piostream;


/**************************************

CLASS
   clString
   
KEYWORDS
   clString

DESCRIPTION
   
  String.h: interface to String utility class
 
  Written by:
  Steven G. Parker
   Department of Computer Science
   University of Utah
   Feb. 1994

  Copyright (C) 1994 SCI Group

PATTERNS
   
WARNING
  
****************************************/
class SCICORESHARE clString {
    struct srep {
	char* s;		// pointer to data
	int n;			// reference count
	srep() { n=1; }
    };
    srep *p;
    clString(int, char*);

public:
    //////////
    // Constructores
    clString();

    //////////
    //<i>No documentation provided</i>
    clString(const clString&);
    //////////
    //<i>No documentation provided</i>
    clString(const char*);

    //////////
    //<i>No documentation provided
    ~clString();

    //////////
    // I/O
    friend SCICORESHARE std::ostream& operator<<(std::ostream& s, const clString&);
    
    //////////
    //<i>No documentation provided</i>
    friend SCICORESHARE std::istream& operator>>(std::istream& s, clString&);

#ifdef DEFINE_OLD_IOSTREAM_OPERATORS
    //////////
    // I/O
    friend SCICORESHARE ::ostream& operator<<(::ostream& s, const clString&);
    
    //////////
    //<i>No documentation provided</i>
    friend SCICORESHARE ::istream& operator>>(::istream& s, clString&);
#endif

    //////////
    //<i>No documentation provided</i>
    clString& operator=(const clString& str);

    
    //////////
    //<i>No documentation provided</i>
    clString& operator=(const char*);
    // Comparison

    //////////
    //Comparison
    int operator==(const char*) const;

    //////////
    //<i>No documentation provided</i>
    int operator==(const clString&) const;

    //////////
    //<i>No documentation provided</i>
    int operator!=(const char*) const;
    
    //////////
    //<i>No documentation provided</i>
    int operator!=(const clString&) const;

    //////////
    //<i>No documentation provided</i>
    int operator<(const char*) const;

    //////////
    //<i>No documentation provided</i>
    int operator<(const clString&) const;

    //////////
    //<i>No documentation provided</i>
    int operator>(const char*) const;

    //////////
    //<i>No documentation provided</i>
    int operator>(const clString&) const;

    //////////
    // Append
    clString operator+(const char*) const;

    
    //////////
    //<i>No documentation provided</i>
    friend SCICORESHARE clString operator+(const char*, const clString&);

    //////////
    //<i>No documentation provided</i>
    clString operator+(const clString&) const;

    
    //////////
    //<i>No documentation provided</i>
    clString& operator+=(const clString&);

    //////////
    //<i>No documentation provided</i>
    clString& operator+=(char);

    //////////
    // Convert to char*
    const char* operator()() const;


    //////////
    // For operating on characters in the string
    // Get ith character
    char operator()(int i) const;
    
    //////////
    // Is ith character alphabetic?
    int is_alpha(int i);
    
    //////////
    // Is ith character a digit?
    int is_digit(int i);
    
    //////////
    // Find where srch appears in the String.  -1 indicates failure.
    int index(const char srch) const;
    
    //////////
    // The length of the string
    int len() const;
    
    //////////
    // A part of the string
    // start=0 is first character
    // length=-1 means to end of string
    clString substr(int start, int length=-1);

    //////////
    // Convert to double/int.  Returns 1 if ok, 0 if bad
    int get_double(double&) const;
    
    //////////
    //<i>No documentation provided</i>
    int get_int(int&) const;

    //////////
    // For the HashTable class
    int hash(int hash_size) const;

    //////////
    // Remove directory name
    friend SCICORESHARE clString basename(const clString&);
    //////////
    // Return directory name
    friend SCICORESHARE clString pathname(const clString&);
  

    static void test_rigorous(RigorousTest* __test);
    static void test_performance(PerfTest* __pt);
};

// Build a string from an int/double
SCICORESHARE clString to_string(int);
SCICORESHARE clString to_string(double);

inline clString::clString()
{
    p=0;
}

inline clString::clString(const clString& str)
{
    if(str.p)str.p->n++;
    p=str.p;
}

inline clString::clString(int, char* s)
{
    p=new srep;
    p->s=s;
}

inline clString::~clString()
{
    if(p && --p->n == 0){
	if(p->s)delete[] p->s;
	delete p;
    }
}

inline clString& clString::operator=(const clString& str)
{
    if(str.p)str.p->n++;		// Protect against st=st
    if(p && --p->n==0){
	if(p->s)delete[] p->s;
	delete p;
    }
    p=str.p;
    return *this;
}

inline const char* clString::operator()() const
{
    return (p && p->s)?p->s:"";
}

} // End namespace Containers
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/10/07 02:07:28  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/09/04 06:01:43  sparker
// Updates to .h files, to minimize #includes
// removed .icc files (yeah!)
//
// Revision 1.2  1999/08/17 06:38:38  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:15  mcq
// Initial commit
//
// Revision 1.5  1999/07/07 21:10:36  dav
// added beginnings of support for g++ compilation
//
// Revision 1.4  1999/05/06 19:55:45  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:34  dav
// added SCICore .h files to /include directories
//
// Revision 1.2  1999/05/03 04:52:17  dmw
// Added and updated DaveW Datatypes/Modules
//
// Revision 1.1.1.1  1999/04/24 23:12:26  dav
// Import sources
//
//

#endif

