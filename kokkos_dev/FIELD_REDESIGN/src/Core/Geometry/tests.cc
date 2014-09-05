#include <SCICore/Tester/TestTable.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>

TestTable test_table[] = {
        {"Vector_Point", &Point::test_rigorous, 0},
	{0,0,0}
};

