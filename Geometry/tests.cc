#include <Tester/TestTable.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>

TestTable test_table[] = {
        {"Point/Vector", &Point::test_rigorous, 0},
	{0,0,0}
};

