
#include "clString.h"
#include "NotFinished.h"
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

clString::srep::srep(char* s)
    : s(s), n(1), left(0), right(0)
{
}

clString::srep::srep(srep* left, srep* right)
    : s(0), n(1), left(left), right(right)
{
    left->n++;
    right->n++;
}

clString::srep::~srep()
{
    if(left){
	left->n--;
	if(left->n==0)
	    delete left;
    }
    if(right){
	right->n--;
	if(right->n==0)
	    delete right;
    }
    if(s)
	delete[] s;
}

clString::clString(srep* rep)
    : rep(rep)
{
}

clString::clString(int, char* c)
{
    rep=new srep(c);
}

void clString::flatten() const
{
    if(rep->s)
	return;
    int l=len();
    char* s=new char[l+1];
    flatten(rep, s);
    rep->s=s;
    if(--rep->left->n == 0)
	delete rep->left;
    if(--rep->right->n == 0)
	delete rep->right;
    rep->left=rep->right=0;
}

void clString::flatten(srep* rep, char* s) const
{
    if(rep->s){
	strcpy(s, rep->s);
    } else {
	flatten(rep->left, s);
	s+=strlen(s);
	flatten(rep->right, s);
    }
}

int clString::len(srep* rep) const
{
    if(rep->s)
	return strlen(rep->s);
    else return len(rep->left)+len(rep->right);
}

clString::clString()
{
    rep=0;
}

clString::clString(const clString& c)
{
    rep=c.rep;
    rep->n++;
}

clString::clString(const char* c)
{
    rep=new srep(strdup(c));
}

clString::~clString()
{
    if(rep){
	rep->n--;
	if(rep->n==0)
	    delete rep;
    }
}

clString& clString::operator=(const clString& str)
{
    if(str.rep != rep) {
	if(rep){
	    rep->n--;
	    if(rep->n==0)
		delete rep;
	}
	rep=str.rep;
	rep->n++;
    }
    return *this;
}

/*
 * I/O
 */
ostream& operator<<(ostream& out, const clString& s) {
    char* p=s();
    if(p)
	out << p;
    return out;
}


/*
 * Comparison
 */
bool clString::operator==(const char* s2) const {
    if(!rep->s)
	flatten();
    return strcmp(rep->s, s2) == 0;
}

bool clString::operator==(const clString& s2) const {
    if(rep==s2.rep)
	return true;
    if(!rep->s)
	flatten();
    if(!s2.rep->s)
	s2.flatten();
    return strcmp(rep->s, s2.rep->s) == 0;
}
bool clString::operator!=(const char* s2) const {
    if(!rep->s)
	flatten();
    return strcmp(rep->s, s2) != 0;
}
bool clString::operator!=(const clString& s2) const {
    if(rep==s2.rep)
	return true;
    if(!rep->s)
	flatten();
    if(!s2.rep->s)
	s2.flatten();
    return strcmp(rep->s, s2.rep->s) != 0;
}
bool clString::operator<(const char* s2) const {
    if(!rep->s)
	flatten();
    return strcmp(rep->s, s2) < 0;
}
bool clString::operator<(const clString& s2) const {
    if(!rep->s)
	flatten();
    if(!s2.rep->s)
	s2.flatten();
    return strcmp(rep->s, s2.rep->s) < 0;
}
bool clString::operator>(const char* s2) const {
    if(!rep->s)
	flatten();
    return strcmp(rep->s, s2) > 0;
}
bool clString::operator>(const clString& s2) const {
    if(!rep->s)
	flatten();
    if(!s2.rep->s)
	s2.flatten();
    return strcmp(rep->s, s2.rep->s) > 0;
}

/*
 * Append
 */
clString clString::operator+(const char* s) const {
    return clString(new srep(rep, new srep(strdup(s))));
}
clString clString::operator+(char c) const {
    char s[2];
    s[0]=c;
    s[1]=0;
    return clString(new srep(rep, new srep(strdup(s))));
}
clString operator+(const char* s1, const clString& s2) {
    return clString(new clString::srep(new clString::srep(strdup(s1)), s2.rep));
}
clString clString::operator+(const clString& s2) const {
    return clString(new srep(rep, s2.rep));
}

clString& clString::operator+=(const char* s) {
    srep* r=new srep(rep, new srep(strdup(s)));
    rep->n--;
    rep=r;
    return *this;
}
clString& clString::operator+=(char c) {
    char s[2];
    s[0]=c;
    s[1]=0;
    srep* r=new srep(rep, new srep(strdup(s)));
    rep->n--;
    rep=r;
    return *this;
}
clString& clString::operator+=(const clString& s2)  {
    srep* r=new srep(rep, s2.rep);
    rep->n--;
    rep=r;
    return *this;
}

/*
 * Convert to char*
 */
char* clString::operator()() const {
    if(!rep->s)
	flatten();
    return rep->s;
}

/*
 * For operating on characters in the string
 * Get ith character
 */
char clString::operator()(int i) const {
    if(!rep->s)
	flatten();
    return rep->s[i];
}
/*
 * Is ith character alphabetic?
 */
int clString::is_alpha(int i) {
    if(!rep->s)
	flatten();
    return isalpha(rep->s[i]);
}

/*
 * Is ith character a digit?
 */
int clString::is_digit(int i) {
    if(!rep->s)
	flatten();
    return isdigit(rep->s[i]);
}
    
/*
 * Find where srch appears in the String.  -1 indicates failure.
 */
int clString::index(const char) const {
    NOT_FINISHED("clString::index");
    return 0;
}

/*
 * Replace all occurrences of <tt>s1</tt> with <tt>s2</tt> and
 * return the new string. Doesn't handle overlapping strings.
 */
clString clString::subs(const clString& s1, const clString& s2) {
    if(!s1.rep->s)
	s1.flatten();
    if(!s2.rep->s)
	s2.flatten();

    if(!rep->s)
	flatten();
    int l=len();
    int l1=s1.len();
    int l2=s2.len();
    char* p=rep->s;
    char* p1=s1.rep->s;
    while(*p){
	if(*p1 == *p) {
	    char* cp1=p1;
	    char* cp=p;
	    while(*cp1 == *cp && *cp1 && *cp){
		cp1++;
		cp++;
		}
	    if(!*cp1){
		// Match...
		l+=l2-l1;
		p=cp-1;
	    }
	}
	p++;
    }
    p=rep->s;
    p1=s1.rep->s;
    char* new_str=new char[l+1];
    char* dp=new_str;
    while(*p){
	if(*p1 == *p) {
	    char* cp1=p1;
	    char* cp=p;
	    while(*cp1 == *cp && *cp1 && *cp){
		cp1++;
		cp++;
		}
	    if(!*cp1){
		cp=p1;
		while(*cp)
		    *dp++=*cp++;
	    } else {
		*dp++=*p++;
	    }
	} else {
	    *dp++=*p++;
	}
    }
    return clString(0, new_str);
}

/*
 * The length of the string
 */
int clString::len() const {
    if(!rep)
	return 0;
    else
	return len(rep);
}
/*
 * A part of the string
 * start=0 is first character
 * length=-1 means to end of string
 */
clString clString::substr(int, int) {
    NOT_FINISHED("clString::substr");
    return clString("");
}

/*
 * Convert to double/int.  Returns true if ok, false if bad
 */
bool clString::get_double(double&) const {
    NOT_FINISHED("clString::get_double");
    return false;
}
bool clString::get_int(int&) const {
    NOT_FINISHED("clString::get_int");
    return false;
}

/*
 * For the HashTable class
 */
unsigned int clString::hash() const {
    unsigned int sum=0;
    if(!rep->s)
	flatten();
    char* p=rep->s;
    while(*p){
	sum=(sum<<3)^(sum>>2)^(*p<<1);
	p++;
    }
    return sum;
}
/*
 *  Build a string from an int/double
 */
clString clString::from(int i) {
    char buf[20];
    sprintf(buf, "%d", i);
    return clString(buf);
}
