/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  String.cc: implementation of String utility class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifdef _WIN32
extern "C" {
size_t strlen(const char*);
char *strcat( char *, const char *);
int strcmp( const char *, const char *);
char *strcpy( char *, const char *);
}
#endif

#include <Core/Containers/String.h>

#include <stdio.h>
#include <iostream>
#include <string.h>
#include <iomanip>
#include <ctype.h>

#include <Core/Util/Assert.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Containers/TrivialAllocator.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Tester/RigorousTest.h>
#include <Core/Tester/PerfTest.h>

namespace SCIRun {

clString& clString::operator=(const char* s)
{
    if(!p || p->n > 1){
	// Disconnect self
	if(p)p->n--;
	p=new srep;
    } else {
	// Free old clString
	if(p->s)delete [] p->s;
    }
    unsigned long len=strlen(s);
    p->s=new char[len+1];
    strcpy(p->s,s);
    return *this;
}

int clString::operator==(const char* s) const
{
    return strcmp((p && p->s)?p->s:"", s) == 0;
}

int clString::operator==(const clString& str) const
{
    return p==str.p || strcmp((p && p->s)?p->s:"", (str.p && str.p->s)?str.p->s:"") == 0;
}

int clString::operator!=(const char* s) const
{
    return strcmp((p && p->s)?p->s:"", s) != 0;
}

int clString::operator!=(const clString& str) const
{
    return p != str.p && strcmp((p && p->s)?p->s:"", (str.p && str.p->s)?str.p->s:"") != 0;
}

int clString::operator<(const char* s) const
{
    return strcmp((p && p->s)?p->s:"", s) < 0;
}

int clString::operator<(const clString& str) const
{
    return p != str.p && strcmp((p && p->s)?p->s:"", (str.p && str.p->s)?str.p->s:"") < 0;
}

int clString::operator>(const char* s) const
{
    return strcmp((p && p->s)?p->s:"", s) > 0;
}

int clString::operator>(const clString& str) const
{
    return p != str.p && strcmp((p && p->s)?p->s:"", (str.p && str.p->s)?str.p->s:"") > 0;
}

char clString::operator()(int index) const
{
    ASSERT(p != 0 && p->s != 0);
    ASSERTRANGE(index, 0, (int)strlen(p->s));
    return p->s[index];
}

int clString::len() const
{
    return p?(int)strlen(p->s):0;
}

int clString::is_alpha(int i)
{
    ASSERT(p != 0 && p->s);
    ASSERTRANGE(i, 0, (int)strlen(p->s));
    return isalpha(p->s[i]);
}

int clString::is_digit(int i)
{
    ASSERT(p != 0);
    ASSERTRANGE(i, 0, (int)strlen(p->s));
    return isdigit(p->s[i]);
}

clString::clString(const char* s)
{
    p=scinew srep;
    unsigned long len=strlen(s);
    p->s=scinew char[len+1];
    strcpy(p->s,s);
}

clString clString::operator+(const clString& str) const
{
    unsigned long newlen=(p?strlen(p->s):0) + (str.p?strlen(str.p->s):0);
    char* ns=scinew char[newlen+1];
    if(p && p->s)strcpy(ns, p->s);
    else ns[0]=0;
    if(str.p && str.p->s)strcat(ns, str.p->s);
    return clString(0, ns);
}

clString clString::operator+(const char* c) const
{
    unsigned long newlen=(p?strlen(p->s):0)+strlen(c);
    char* ns=scinew char[newlen+1];
    if(p && p->s)strcpy(ns, p->s);
    else ns[0]=0;
    strcat(ns, c);
    return clString(0, ns);
}

clString operator+(const char* c, const clString& str)
{
    unsigned long newlen=(str.p?strlen(str.p->s):0)+strlen(c);
    char* ns=scinew char[newlen+1];
    strcpy(ns, c);
    if(str.p && str.p->s)strcat(ns, str.p->s);
    return clString(0, ns);
}

int clString::get_double(double& x) const
{
    return (p && p->s)?sscanf(p->s, "%lf", &x)==1:0;
}

int clString::get_int(int& x) const
{
    return (p && p->s)?sscanf(p->s, "%d", &x)==1:0;
}

clString to_string(int n)
{
    char s[50];
    sprintf(s,"%d",n);
    return clString(s);
}

clString to_string(double d)
{
    char s[50];
    sprintf(s,"%g",d);
    return clString(s);
}

std::ostream& operator<<(std::ostream& s, const clString& str)
{
    return s << ((str.p && str.p->s)?str.p->s:"");
}

std::istream& operator>>(std::istream& s, clString& str)
{
  const int bufsize = 1024;
  char* buf = scinew char[bufsize];
  s >> ::std::setw( bufsize ) >> buf;
  str = buf;
  while (s.gcount() == bufsize - 1) { // string is longer than 1024
    s >> ::std::setw( bufsize) >> buf;
    str +=  buf;
  }
  delete [] buf;
  return s;
}

#if defined(__sgi) && !defined(__GNUC__)
} // End namespace SCIRun

#include <iostream>
#include <iomanip.h>

namespace SCIRun {

::ostream& operator<<(::ostream& s, const clString& str)
{
    return s << str();
}

::istream& operator>>(::istream& s, clString& str)
{
  const int bufsize = 1024;
  char* buf = scinew char[bufsize];
  s >> ::setw( bufsize ) >> buf;
  str = buf;
  while (s.gcount() == bufsize - 1) { // string is longer than 1024
    s >> ::setw( bufsize) >> buf;
    str +=  buf;
  }
  delete [] buf;
  return s;
}
#endif


int clString::index(const char match) const
{
    if(!p || !p->s)return -1;
    int i=0;
    char* pp=p->s;
    while(*pp){
	if(*pp == match)return i;
	i++;
	pp++;
    }
    return -1;
}

clString clString::substr(int start, int length) const
{
    ASSERT(p != 0);
    unsigned long len=strlen(p->s);
    ASSERTRANGE(start, 0, (int)len);
    unsigned long l=length==-1?len-start:length;
    ASSERTRANGE((long)(start+l), 0, (long)(len+1));
    char* tmp=scinew char[l+1];
    int i;
    for(i=0;i<(int)l;i++){
	tmp[i]=p->s[i+start];
    }
    tmp[i]='\0';
    clString rstr(0, tmp);
    return rstr;
}

int clString::hash(int hash_size) const
{
    if(!p || !p->s)return 0;
    unsigned char* pp=(unsigned char*)p->s;
    int sum=0;
    while(*pp){
	sum=(sum << 2) ^ (sum >> 2) ^ *pp;
	pp++;
    }
    sum=sum<0?-sum:sum;
    return sum%hash_size;
}

clString basename(const clString& str)
{
    ASSERT(str.p && str.p->s);
    char* pp=str.p->s;
    char* last_slash=pp;
    while(*pp){
	if(*pp=='/')last_slash=pp;
	pp++;
    }
    return clString(last_slash+1);
}

clString pathname(const clString &str)
{
  ASSERT(str.p && str.p->s);
  char* pp=str.p->s;
  char* p = pp;
  char* last_slash = pp;
  clString result;
  while(*pp) {
   if(*pp=='/')last_slash=pp;
	pp++;
    }
  
  while( p != last_slash ) {
    result += *p;
    p++;
  }
  return result;
}

clString& clString::operator+=(char c)
{
    if(p){
	if(p->n != 1){
	    // detach...
	    srep* oldp=p;
	    p=scinew srep;
	    unsigned long len=strlen(oldp->s);
	    p->s=scinew char[len+2];
	    strcpy(p->s, oldp->s);
	    p->s[len]=c;
	    p->s[len+1]=0;
	    oldp->n--;
	    p->n=1;
	} else {
	    char* olds=p->s;
	    unsigned long len=strlen(olds);
	    p->s=scinew char[len+2];
	    strcpy(p->s, olds);
	    p->s[len]=c;
	    p->s[len+1]=0;
	    delete[] olds;
	}
    } else {
	p=scinew srep;
	p->n=1;
	p->s=scinew char[2];
	p->s[0]=c;
	p->s[1]=0;
    }
    return *this;
}

clString& clString::operator+=(const clString& str)
{
    unsigned long newlen=(p?strlen(p->s):0)+(str.p?strlen(str.p->s):0);
    char* ns=scinew char[newlen+1];
    if(p && p->s)strcpy(ns, p->s);
    else ns[0]=0;
    if(str.p && str.p->s)strcat(ns, str.p->s);
    if(p && p->n > 1){
	if(p)p->n--;
	p=scinew srep;
    } else {
	if(p && p->s)delete[] p->s;
	if(!p)
	    p=scinew srep;
    }
    p->s=ns;
    return *this;
}

void clString::test_rigorous(RigorousTest* __test)
{
    clString s0("hi");
    clString s1(s0+" ");
    clString s2("there");
    TEST(s0<s2);
    TEST(s2>s0);
    clString s3(s1+s2);
    TEST(s3=="hi there");
    clString s4(s3);
    TEST(s1 != s4);
    clString s5(s4());
    TEST(s5==s4);
    TEST(s5.len()==8);
    TEST(s3(0)=='h');
    TEST(s3(2)==' ');
    TEST(s3(7)=='e');
    s3="";
    int i;
    int n=8;
    for(i=0;i<n;i++){
	s3=s3+"0123456789";
    }
    for(i=0;i<n;i++){
	s3=s3+"0123456789";
    }
    n=n+n;
    for(i=0;i<3;i++){
	s3=s3+s3;
    }
    n<<=3;
    for(i=0;i<3;i++){
	s3+=s3;
    }
    n<<=3;
    TEST(s3.len()==n*10);
    int idx=0;
    for(i=0;i<n;i++){
	for(int j=0;j<10;j++){
	    TEST(s3(idx++)==j+'0');
	}
    }

    

}

/*
 * Performance tests
 */
void clString::test_performance(PerfTest* __pt) {
    PERFTEST("baseline") {
    }

    /*
     * Test CTORs
     */
    PERFTEST("CTOR/DTOR - empty") {
	clString str;
    }
    PERFTEST("CTOR/DTOR - with small string") {
	clString str("asdf");
    }
    static char* large_string="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    PERFTEST("CTOR/DTOR - with large string") {
	clString str(large_string);
    }
    clString s1("asdf");
    PERFTEST("CTOR - copy small string") {
	clString str(s1);
    }
    clString s2(large_string);
    PERFTEST("CTOR - copy large string") {
	clString str(s2);
    }
    
    /*
     * Various operators.
     */
#if 0
    clString s3;
    PERFTEST("operator= - small string from clString") {
	s3=s1;
    }
    clString s4;
    PERFTEST("operator= - large string from clString") {
	s4=s2;
    }
    PERFTEST("operator= - small string from char*") {
	s3="asdf";
    }
    PERFTEST("operator= - large string from char*") {
	s4=large_string;
    }
    PERFTEST("operator+ - pairwise") {
	s4=s1+s2;
    }
#endif
    PERFTEST("compound + - big") {
	clString str(s2+s2+s2+s2+s2+s2+s2+s2+s2+s2+s2+s2+s2+s2+s2+s2);
    }
    PERFTEST("compound + - flatten big") {
	clString str(s2+s2+s2+s2+s2+s2+s2+s2+s2+s2+s2+s2+s2+s2+s2+s2);
	str();
    }
}

} // End namespace SCIRun
