
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

#ifndef SCI_Classlib_BitArray1_h
#define SCI_Classlib_BitArray1_h 1

class BitArray1 {
    int size;
    int nbits;
    unsigned char* bits;
public:
    BitArray1(int size, int initial);
    ~BitArray1();
    int is_set(int);
    void set(int);
    void clear(int);

    void clear_all();
    void set_all();
};

#endif

