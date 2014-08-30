/*
 * Copyright (c) 2014 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <spatialops/structured/GhostData.h>
#include <spatialops/structured/MemoryWindow.h>
#include <test/TestHelper.h>

#include <iostream>

using namespace SpatialOps;
using namespace std;

int main()
{
  TestHelper overall(true);

  {
    TestHelper status(false);
    IntVec a( 3,4,5 );
    IntVec b( 1,4,2 );
    IntVec c = b+a;
    status( c == IntVec(4,8,7) );
    c = b-a;
    status( c == IntVec(-2,0,-3) );
    c += a;
    status( c == b );
    c -= b;
    status( c == IntVec(0,0,0) );
    overall( status.ok(), "Simple IntVec operations" );
  }

  {
    TestHelper status(false);
    MemoryWindow w( IntVec(2,2,2) );
    status( 0 == w.flat_index( IntVec(0,0,0) ), "(0,0,0)" );
    status( 1 == w.flat_index( IntVec(1,0,0) ), "(1,0,0)" );
    status( 2 == w.flat_index( IntVec(0,1,0) ), "(0,1,0)" );
    status( 3 == w.flat_index( IntVec(1,1,0) ), "(1,1,0)" );
    status( 4 == w.flat_index( IntVec(0,0,1) ), "(0,0,1)" );
    status( 5 == w.flat_index( IntVec(1,0,1) ), "(1,0,1)" );
    status( 6 == w.flat_index( IntVec(0,1,1) ), "(0,1,1)" );
    status( 7 == w.flat_index( IntVec(1,1,1) ), "(1,1,1)" );
    overall( status.ok(), "2x2x2 global patch no subdivisions" );
  }
  {
    TestHelper status(false);
    IntVec globDim(4,4,4), extent(2,2,2);

    MemoryWindow w000( globDim, IntVec(0,0,0), extent );
    status( 0 == w000.flat_index( IntVec(0,0,0) ), "000: (0,0,0)" );
    status( 1 == w000.flat_index( IntVec(1,0,0) ), "000: (1,0,0)" );
    status( 4 == w000.flat_index( IntVec(0,1,0) ), "000: (0,1,0)" );
    status( 5 == w000.flat_index( IntVec(1,1,0) ), "000: (1,1,0)" );
    status( 16== w000.flat_index( IntVec(0,0,1) ), "000: (0,0,1)" );
    status( 17== w000.flat_index( IntVec(1,0,1) ), "000: (1,0,1)" );
    status( 20== w000.flat_index( IntVec(0,1,1) ), "000: (0,1,1)" );
    status( 21== w000.flat_index( IntVec(1,1,1) ), "000: (1,1,1)" );

    MemoryWindow w100( globDim, IntVec(2,0,0), extent );
    status( 2 == w100.flat_index( IntVec(0,0,0) ), "100: (0,0,0)" );
    status( 3 == w100.flat_index( IntVec(1,0,0) ), "100: (1,0,0)" );
    status( 6 == w100.flat_index( IntVec(0,1,0) ), "100: (0,1,0)" );
    status( 7 == w100.flat_index( IntVec(1,1,0) ), "100: (1,1,0)" );
    status( 18== w100.flat_index( IntVec(0,0,1) ), "100: (0,0,1)" );
    status( 19== w100.flat_index( IntVec(1,0,1) ), "100: (1,0,1)" );
    status( 22== w100.flat_index( IntVec(0,1,1) ), "100: (0,1,1)" );
    status( 23== w100.flat_index( IntVec(1,1,1) ), "100: (1,1,1)" );

    MemoryWindow w010( globDim, IntVec(0,2,0), extent );
    status( 8 == w010.flat_index( IntVec(0,0,0) ), "010: (0,0,0)" );
    status( 9 == w010.flat_index( IntVec(1,0,0) ), "010: (1,0,0)" );
    status( 12== w010.flat_index( IntVec(0,1,0) ), "010: (0,1,0)" );
    status( 13== w010.flat_index( IntVec(1,1,0) ), "010: (1,1,0)" );
    status( 24== w010.flat_index( IntVec(0,0,1) ), "010: (0,0,1)" );
    status( 25== w010.flat_index( IntVec(1,0,1) ), "010: (1,0,1)" );
    status( 28== w010.flat_index( IntVec(0,1,1) ), "010: (0,1,1)" );
    status( 29== w010.flat_index( IntVec(1,1,1) ), "010: (1,1,1)" );

    MemoryWindow w110( globDim, IntVec(2,2,0), extent );
    status( 10== w110.flat_index( IntVec(0,0,0) ), "110: (0,0,0)" );
    status( 11== w110.flat_index( IntVec(1,0,0) ), "110: (1,0,0)" );
    status( 14== w110.flat_index( IntVec(0,1,0) ), "110: (0,1,0)" );
    status( 15== w110.flat_index( IntVec(1,1,0) ), "110: (1,1,0)" );
    status( 26== w110.flat_index( IntVec(0,0,1) ), "110: (0,0,1)" );
    status( 27== w110.flat_index( IntVec(1,0,1) ), "110: (1,0,1)" );
    status( 30== w110.flat_index( IntVec(0,1,1) ), "110: (0,1,1)" );
    status( 31== w110.flat_index( IntVec(1,1,1) ), "110: (1,1,1)" );

    MemoryWindow w001( globDim, IntVec(0,0,2), extent );
    status( 32== w001.flat_index( IntVec(0,0,0) ), "001: (0,0,0)" );
    status( 33== w001.flat_index( IntVec(1,0,0) ), "001: (1,0,0)" );
    status( 36== w001.flat_index( IntVec(0,1,0) ), "001: (0,1,0)" );
    status( 37== w001.flat_index( IntVec(1,1,0) ), "001: (1,1,0)" );
    status( 48== w001.flat_index( IntVec(0,0,1) ), "001: (0,0,1)" );
    status( 49== w001.flat_index( IntVec(1,0,1) ), "001: (1,0,1)" );
    status( 52== w001.flat_index( IntVec(0,1,1) ), "001: (0,1,1)" );
    status( 53== w001.flat_index( IntVec(1,1,1) ), "001: (1,1,1)" );

    MemoryWindow w101( globDim, IntVec(2,0,2), extent );
    status( 34== w101.flat_index( IntVec(0,0,0) ), "101: (0,0,0)" );
    status( 35== w101.flat_index( IntVec(1,0,0) ), "101: (1,0,0)" );
    status( 38== w101.flat_index( IntVec(0,1,0) ), "101: (0,1,0)" );
    status( 39== w101.flat_index( IntVec(1,1,0) ), "101: (1,1,0)" );
    status( 50== w101.flat_index( IntVec(0,0,1) ), "101: (0,0,1)" );
    status( 51== w101.flat_index( IntVec(1,0,1) ), "101: (1,0,1)" );
    status( 54== w101.flat_index( IntVec(0,1,1) ), "101: (0,1,1)" );
    status( 55== w101.flat_index( IntVec(1,1,1) ), "101: (1,1,1)" );

    MemoryWindow w011( globDim, IntVec(0,2,2), extent );
    status( 40== w011.flat_index( IntVec(0,0,0) ), "011: (0,0,0)" );
    status( 41== w011.flat_index( IntVec(1,0,0) ), "011: (1,0,0)" );
    status( 44== w011.flat_index( IntVec(0,1,0) ), "011: (0,1,0)" );
    status( 45== w011.flat_index( IntVec(1,1,0) ), "011: (1,1,0)" );
    status( 56== w011.flat_index( IntVec(0,0,1) ), "011: (0,0,1)" );
    status( 57== w011.flat_index( IntVec(1,0,1) ), "011: (1,0,1)" );
    status( 60== w011.flat_index( IntVec(0,1,1) ), "011: (0,1,1)" );
    status( 61== w011.flat_index( IntVec(1,1,1) ), "011: (1,1,1)" );

    MemoryWindow w111( globDim, IntVec(2,2,2), extent );
    status( 42== w111.flat_index( IntVec(0,0,0) ), "111: (0,0,0)" );
    status( 43== w111.flat_index( IntVec(1,0,0) ), "111: (1,0,0)" );
    status( 46== w111.flat_index( IntVec(0,1,0) ), "111: (0,1,0)" );
    status( 47== w111.flat_index( IntVec(1,1,0) ), "111: (1,1,0)" );
    status( 58== w111.flat_index( IntVec(0,0,1) ), "111: (0,0,1)" );
    status( 59== w111.flat_index( IntVec(1,0,1) ), "111: (1,0,1)" );
    status( 62== w111.flat_index( IntVec(0,1,1) ), "111: (0,1,1)" );
    status( 63== w111.flat_index( IntVec(1,1,1) ), "111: (1,1,1)" );

    overall( status.ok(), "4x4x4 global divided into 8 2x2x2 patches");
  }

  {
    TestHelper status(false);
    IntVec globDim(4,4,4), extent(2,4,4);

    MemoryWindow w1( globDim, IntVec(0,0,0), extent );
    status( 0 == w1.flat_index( IntVec(0,0,0) ), "1: (0,0,0)" );
    status( 1 == w1.flat_index( IntVec(1,0,0) ), "1: (1,0,0)" );
    status( 4 == w1.flat_index( IntVec(0,1,0) ), "1: (0,1,0)" );
    status( 5 == w1.flat_index( IntVec(1,1,0) ), "1: (1,1,0)" );
    status( 16== w1.flat_index( IntVec(0,0,1) ), "1: (0,0,1)" );
    status( 17== w1.flat_index( IntVec(1,0,1) ), "1: (1,0,1)" );
    status( 21== w1.flat_index( IntVec(1,1,1) ), "1: (1,1,1)" );
    status( 37== w1.flat_index( IntVec(1,1,2) ), "1: (3,0,1)" );

    overall( status.ok(), "4,4,4 global divided into 2 2x4x4 patches" );
  }

  {
    TestHelper status(false);
    IntVec globDim(4,4,4), offset(2,2,2), extent(1,1,1);
    const MemoryWindow w( globDim, extent, offset );
    status( 21 == w.flat_index( IntVec(0,0,0) ), "(0,0,0)" );
    status( 42 == w.flat_index( IntVec(1,1,1) ), "(1,1,1)" );
    status( 8 == w.local_npts(), "#points interior" );
    overall( status.ok(), "2x2x2 interior of a 4x4x4 box" );
  }

  if( overall.isfailed() ) return -1;
  return 0;
}
