
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


#include <Classlib/String.h>
#include <Classlib/Assert.h>
#include <Classlib/Persistent.h>
#include <iostream.h>
#include <stdio.h>
#include <string.h>
#define inline
#include <Classlib/String.icc>
#undef inline

clString::clString(const char* s)
{
    p=new srep;
    p->len=strlen(s);
    p->s=new char[p->len+1];
    strcpy(p->s,s);
}

clString clString::operator+(const clString& str) const
{
    int newlen=(p?p->len:0) + (str.p?str.p->len:0);
    char* ns=new char[newlen+1];
    if(p && p->s)strcpy(ns, p->s);
    else ns[0]=0;
    if(str.p && str.p->s)strcat(ns, str.p->s);
    return clString(0, ns);
}

clString clString::operator+(const char* c) const
{
    int newlen=(p?p->len:0)+strlen(c);
    char* ns=new char[newlen+1];
    if(p && p->s)strcpy(ns, p->s);
    else ns[0]=0;
    strcat(ns, c);
    return clString(0, ns);
}

clString operator+(const char* c, const clString& str)
{
    int newlen=(str.p?str.p->len:0)+strlen(c);
    char* ns=new char[newlen+1];
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

ostream& operator<<(ostream& s, const clString& str)
{
    return s << ((str.p && str.p->s)?str.p->s:"");
}

istream& operator>>(istream& s, clString& str)
{
    char* buf=new char[1000];
    s.get(buf,1000,'\n');
#ifdef broken
    char c;
    if(cin.get(c) && c!='\n'){
	// Longer than 1000...
	int grow=1;
	int size=1000;
	while(grow){
	    int newsize=size << 1; /* Double size... */
	    char* p=new char[newsize];
	    strncpy(p, buf, size);
	    s.get(buf+size,size,'\n');
	    if(cin.get(c) && c!='\n'){
		grow=1;
	    } else {
		grow=0;
	    }
	    delete[] buf;
	    buf=p;
	    size=newsize;
	}
    }
#endif
    str=buf; // Uses operator=
    delete[] buf;
    return s;
}

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

clString clString::substr(int start, int length)
{
    ASSERT(p != 0);
    ASSERTRANGE(start, 0, p->len);
    int l=length==-1?p->len-start:length;
    ASSERTRANGE(start+l, 0, p->len+1);
    char* tmp=new char[l+1];
    for(int i=0;i<l;i++){
	tmp[i]=p->s[i+start];
    }
    tmp[i]='\0';
    clString rstr(0, tmp);
    return rstr;
}

int clString::hash(int hash_size) const
{
    if(!p || !p->s)return 0;
    char* pp=p->s;
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

clString& clString::operator+=(char c)
{
    if(p){
	if(p->n != 1){
	    // detach...
	    srep* oldp=p;
	    p=new srep;
	    p->s=new char[p->len+2];
	    strcpy(p->s, oldp->s);
	    p->s[oldp->len]=c;
	    p->s[oldp->len+1]=0;
	    p->len=oldp->len+1;
	    oldp->n--;
	    p->n=1;
	} else {
	    char* olds=p->s;
	    p->s=new char[p->len+2];
	    strcpy(p->s, olds);
	    p->s[p->len]=c;
	    p->s[p->len+1]=0;
	    p->len++;
	}
    } else {
	p=new srep;
	p->n=1;
	p->s=new char[2];
	p->s[0]=c;
	p->s[1]=0;
	p->len=1;
    }
    return *this;
}

clString& clString::operator+=(const clString& str)
{
    int newlen=(p?p->len:0)+(str.p?str.p->len:0);
    char* ns=new char[newlen+1];
    if(p && p->s)strcpy(ns, p->s);
    else ns[0]=0;
    if(str.p && str.p->s)strcat(ns, str.p->s);
    if(p && p->n > 1){
	if(p)p->n--;
	p=new srep;
    } else {
	if(p && p->s)delete[] p->s;
	if(!p)
	    p=new srep;
    }
    p->len=newlen;
    p->s=ns;
    return *this;
}
