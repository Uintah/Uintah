//static char *id="@(#) $Id$";

/*
 *  BitArray1.cc: 1D bit array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Containers/BitArray1.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Tester/RigorousTest.h>

#include <iostream.h>

namespace SCICore {
namespace Containers {

BitArray1::BitArray1(int size, int initial)
  : size(size), nbits(size/8 + ((size&7)?1:0)), bits(scinew unsigned char)//[nbits])
{
    if(initial)
	set_all();
    else
	clear_all();
}

BitArray1::~BitArray1()
{
    delete[] bits;
}

int BitArray1::is_set(int ix)
{
    int b=ix>>3;
    int i=ix&7;
    int mask=1<<i;
    return (bits[b]&mask) != 0;
}

void BitArray1::set(int ix)
{
    int b=ix>>3;
    int i=ix&7;
    int mask=1<<i;
    bits[b]|=mask;
}

void BitArray1::clear(int ix)
{
    int b=ix>>3;
    int i=ix&7;
    int mask=1<<i;
    bits[b]&=~mask;
}

void BitArray1::clear_all()
{
    for(int i=0;i<nbits;i++)
	bits[i]=0;
}

void BitArray1::set_all()
{
    for(int i=0;i<nbits;i++)
	bits[i]=0xff;
}

void BitArray1::test_rigorous(RigorousTest* __test)
{
    BitArray1 b(10000,0);
    int i;

    for(i=0;i<10000;i++){
	TEST(b.is_set(i)==0);
    }
    
    for(i=0;i<10000;i+=2)
	b.set(i);

    for(i=0;i<10000;i+=2)
	TEST(b.is_set(i));

    b.clear_all();
    for(i=0;i<10000;i++)
	TEST(!b.is_set(i));

    for(i=0;i<10000;i++)
	TEST(!b.is_set(i));

    for(i=0;i<10000;i+=5)
	b.set(i);

    for(i=0;i<10000;i+=2)
	b.set(i);

    for(i=0;i<10000;i+=5)
	TEST(b.is_set(i));

    for(i=0;i<10000;i+=2)
	TEST(b.is_set(i));

    b.set_all();

    for(i=0;i<10000;i++)
	TEST(b.is_set(i));

    for(i=0;i<10000;i+=2)
	b.clear(i);
    for(i=0;i<10000;i+=5)
	b.clear(i);

    for(i=0;i<10000;i+=2)
	TEST(!b.is_set(i));
    for(i=0;i<10000;i+=5)
	TEST(!b.is_set(i));

    b.clear_all();
    for(i=0;i<10000;i++)
	TEST(!b.is_set(i));
}

} // End namespace Containers
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:35  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:11  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:26  dav
// Import sources
//
//
