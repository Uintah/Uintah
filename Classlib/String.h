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

class ostream;
class istream;
class Piostream;

#ifdef __GNUG__
#pragma interface
#endif

class clString {
    struct srep {
	char* s;		// pointer to data
	int n;			// reference count
	int len;
	srep() { n=1; }
    };
    srep *p;
    clString(int, char*);
public:
    // Constructores
    clString();
    clString(const clString&);
    clString(const char*);
    ~clString();

    // I/O
    friend ostream& operator<<(ostream& s, const clString&);
    friend istream& operator>>(istream& s, clString&);

    // Comparison
    clString& operator=(const clString& str);
    clString& operator=(const char*);
    int operator==(const char*) const;
    int operator==(const clString&) const;
    int operator!=(const char*) const;
    int operator!=(const clString&) const;
    int operator<(const char*) const;
    int operator<(const clString&) const;
    int operator>(const char*) const;
    int operator>(const clString&) const;

    // Append
    clString operator+(const char*) const;
    friend clString operator+(const char*, const clString&);
    clString operator+(const clString&) const;
    clString& operator+=(const char*);
    clString& operator+=(const clString&);
    clString& operator+=(char);

    // Convert to char*
    char* operator()() const;

    // For operating on characters in the string
    // Get ith character
    char operator()(int i) const;
    // Is ith character alphabetic?
    int is_alpha(int i);
    // Is ith character a digit?
    int is_digit(int i);
    // Find where srch appears in the String.  -1 indicates failure.
    int index(const char srch) const;
    // The length of the string
    int len() const;
    // A part of the string
    // start=0 is first character
    // length=-1 means to end of string
    clString substr(int start, int length=-1);

    // Convert to double/int.  Returns 1 if ok, 0 if bad
    int get_double(double&) const;
    int get_int(int&) const;

    // For the HashTable class
    int hash(int hash_size) const;

    // Remove directory name
    friend clString basename(const clString&);
};

// Build a string from an int/double
clString to_string(int);
clString to_string(double);
clString basename(const clString&);

#endif
