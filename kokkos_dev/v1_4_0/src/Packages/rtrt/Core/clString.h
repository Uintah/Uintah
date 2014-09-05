/*
 *  String.h: interface to String utility class
 *  <pre>
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 *  </pre>
 */

//class ostream;
//class istream;
#include <iostream>
using namespace std;

class Piostream;

class clString {
    struct srep {
	char* s;		// pointer to data
	int n;			// reference count
	srep* left;
	srep* right;
	srep(char* s);
	srep(srep* left, srep* right);
	~srep();
    };
    srep *rep;
    clString(srep* rep);
    clString(int, char* c);
    void flatten() const;
    void flatten(srep* rep, char* s) const;
    int len(srep* rep) const;
public:
    /*
     * Constructors
     */
    clString();
    clString(const clString& c);
    clString(const char* c);
    ~clString();
    clString& operator=(const clString& str);

    /*
     * I/O
     */
    friend ostream& operator<<(ostream& out, const clString& s);
    friend istream& operator>>(istream& s, clString&);

    /*
     * Comparison
     */
    bool operator==(const char* s2) const;
    bool operator==(const clString& s2) const;
    bool operator!=(const char* s2) const;
    bool operator!=(const clString& s2) const;
    bool operator<(const char* s2) const;
    bool operator<(const clString& s2) const;
    bool operator>(const char* s2) const;
    bool operator>(const clString& s2) const;

    /*
     * Append
     */
    clString operator+(const char* s) const;
    clString operator+(char c) const;
    friend clString operator+(const char* s1, const clString& s2);
    clString operator+(const clString& s2) const;
    clString& operator+=(const char* s);
    clString& operator+=(char c);
    clString& operator+=(const clString& s2);

    /*
     * Convert to char*
     */
    char* operator()() const;

    /*
     * For operating on characters in the string
     * Get ith character
     */
    char operator()(int i) const;

    /*
     * Is ith character alphabetic?
     */
    int is_alpha(int i);

    /*
     * Is ith character a digit?
     */
    int is_digit(int i);
    
    /*
     * Find where srch appears in the String.  -1 indicates failure.
     */
    int index(const char srch) const;

    /*
     * Replace all occurrences of <tt>s1</tt> with <tt>s2</tt> and
     * return the new string. Doesn't handle overlapping strings.
     */
     clString subs(const clString& s1, const clString& s2);

    /*
     * The length of the string
     */
    int len() const;

    /*
     * A part of the string
     * start=0 is first character
     * length=-1 means to end of string
     */
    clString substr(int start, int length=-1);

    /*
     * Convert to double/int.  Returns true if ok, false if bad
     */
    bool get_double(double&) const;
    bool get_int(int&) const;

    /*
     * For the HashTable class
     */
    unsigned int hash() const;

    /*
     *  Build a string from an int/double
     */
    static clString from(int i);
    static clString from(double);

};
