#include <Tester/TestTable.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Geometry/BBox.h>

TestTable test_table[] = {
  //{"Point/Vector", &Point::test_rigorous, 0},
       	{"BBox",&BBox::test_rigorous,0},
	{0,0,0}
};

