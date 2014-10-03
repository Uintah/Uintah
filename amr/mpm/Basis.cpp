#include"Basis.h"
void Basis::Print(const string& msg)
{
	cout << endl << "Printing information for basis function " << msg << endl;
	cout << "origin is: (" << org[x1] << ", " << org[x2] << ")" << endl;
	cout << "(dx, dy) is: (" << dx << ", " << dy << ")" << endl;
}

double Basis::Eval(const Vec2D& v)
{
	Vec2D dv = v - org;
	double value = (1.0 - dv[x1]/dx)*(1.0 - dv[x2]/dy);
	return value;
}

double Basis::dxEval(const Vec2D& v)
{
	return 1.0/dx;
}

double Basis::dyEval(const Vec2D& v)
{
	return 1.0/dy;
}

Vec2D Basis::Grad(const Vec2D& v)
{
	Vec2D result = {1.0/dx, 1.0/dy};
	return result;
}
