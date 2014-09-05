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

#include <Core/Containers/BitArray1.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Tester/RigorousTest.h>

namespace SCIRun {

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

} // End namespace SCIRun

