// Main header for the -*- C++ -*- string classes.

#ifndef __UTIL_STRING__
#define __UTIL_STRING__


#include <stdio.h>

#include <string>

using namespace std;

namespace SemotusVisum {

/**
 * Compares strings without regard to case.
 *
 * @param s1    String 1
 * @param s2    String 2
 * @return  0 if equal, -1 if s1 < s2, 1 if s1 > s2
 */
int strcasecmp( const string &s1, const string &s2 );

/**
 * Creates a string from the given unsigned int
 *
 * @param i     Value to be string-ized
 * @return      String representation of the parameter
 */
string mkString( const unsigned int i );

/**
 * Creates a string from the given int
 *
 * @param i     Value to be string-ized
 * @return      String representation of the parameter
 */
string mkString( const int i );

/**
 * Creates a string from the given double
 *
 * @param d     Value to be string-ized
 * @return      String representation of the parameter
 */
string mkString( const double d );

/**
 * Creates a string from the given pointer
 *
 * @param v     Value to be string-ized
 * @return      String representation of the parameter
 */
string mkString( const void * v );


/**
 * Converts the string to a newly allocated (user-deleted) char *.
 *
 * @param s     String to convert
 * @return      Modifyable char * that the user must delete.
 */
char * toChar( const string &s );


/**
 * Converts the string to an integer
 *
 * @param s     String to convert
 * @return      Integer value in s, if any
 */
int atoi( const string& s );

/**
 * Converts the string to an double
 *
 * @param s     String to convert
 * @return      Double value in s, if any
 */
double atof( const string& s );

/**
 * Returns true if haystack starts with needle
 *
 * @param haystack     Large string
 * @param needle       Prefix to test for
 * @return             True if the haystack if prefixed with the needle
 */
bool startsWith( const string &haystack, const string &needle );

/**
 * Returns true if haystack ends with needle
 *
 * @param haystack     Large string
 * @param needle       Suffix to test for
 * @return             True if the haystack if suffixed with the needle
 */
bool endsWith( const string &haystack, const string &needle );


// Macros to get file:line
#define __util_str__(x) #x
#define __util_xstr__(x) __util_str__(x)
#define __HERE__ (  __FILE__ ":" __util_xstr__(__LINE__)  )
}

#endif
