#include <Tester/TestTable.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>

TestTable test_table[] = {
        {"Vector_Point", &Point::test_rigorous, 0},
	{0,0,0}
};

