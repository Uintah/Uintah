

/*
 *  BitArray1.h: 1D bit array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/BitArray1.h>

BitArray1::BitArray1(int size, int initial)
: size(size), nbits(size/8), bits(new unsigned char[nbits])
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

