#ifndef SCI_Wangxl_Datatypes_Mesh_Utilities_h
#define SCI_Wangxl_Datatypes_Mesh_Utilities_h

#include <assert.h>

namespace Wangxl {

  //using namespace SCIRun;

struct Utilities
{
  static const char tab_next_around_edge[4][4];

  static int ccw(int i)
    {
      assert( i >= 0 && i < 3 );
      return (i==2) ? 0 : i+1;
    }
  
  static int cw(int i)
    {
      assert( i >= 0 && i < 3 );
      return (i==0) ? 2 : i-1;
    }

  static int next_around_edge(const int i, const int j)
  {
    // index of the next cell when turning around the
    // oriented edge vertex(i) vertex(j) in 3d
    assert( (i >= 0 && i < 4) && (j >= 0 && j < 4) && (i != j) );
    return tab_next_around_edge[i][j];
  }

  static unsigned int random_value, count, val;

  // rand_4() outputs pseudo random unsigned ints < 4.
  // We compute random 16 bit values, that we slice/shift to make it faster.
  static unsigned int rand_4()
  {
      if (count==0)
      {
          count = 16;
          random_value = (421 * random_value + 2073) % 32749;
          val = random_value;
      }
      count--;
      unsigned int ret = val & 3;
      val = val >> 1;
      return ret;
  }

  static unsigned int rand_3()
  {
      unsigned int i = rand_4();
      return i==3 ? 0 : i;
  }
};

}

#endif
