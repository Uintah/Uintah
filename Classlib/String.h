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

#ifndef SCI_Classlib_String_h
#define SCI_Classlib_String_h 1

#include <stdlib.h> // For size_t

class ostream;
class istream;
class Piostream;
class RigorousTest;
class PerfTest;

#ifdef __GNUG__
#pragma interface
#endif



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
class clString {
    struct srep {
	char* s;		// pointer to data
	int n;			// reference count
	srep() { n=1; }
// Can't do this yet - broken
//	inline void* operator new(size_t);
//	inline void operator delete(void*, size_t);
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
    friend ostream& operator<<(ostream& s, const clString&);
    
    //////////
    //<i>No documentation provided</i>
    friend istream& operator>>(istream& s, clString&);


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
    friend clString operator+(const char*, const clString&);

    //////////
    //<i>No documentation provided</i>
    clString operator+(const clString&) const;

    
    //////////
    //<i>No documentation provided</i>
    clString& operator+=(const char*);

    //////////
    //<i>No documentation provided</i>
    clString& operator+=(const clString&);

    //////////
    //<i>No documentation provided</i>
    clString& operator+=(char);

    //////////
    // Convert to char*
    char* operator()() const;


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
    friend clString basename(const clString&);

    static void test_rigorous(RigorousTest* __test);
    static void test_performance(PerfTest* __pt);
};

// Build a string from an int/double
clString to_string(int);
clString to_string(double);

#endif









