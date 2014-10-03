#include"Utils.h"

/////////////////////////////////////////////////////////////////////////////////
//vector operators
/////////////////////////////////////////////////////////////////////////////////
Vec2D operator+(const Vec2D& a, const Vec2D& b)
{
	Vec2D c = {a[x1] + b[x1], a[x2] + b[x2]};
	return c;
}

void operator+=(Vec2D& a, const Vec2D& b)
{
	a = {a[x1] + b[x1], a[x2] + b[x2]};
}

Vec2D operator-(const Vec2D& a, const Vec2D& b)
{
	Vec2D c = {a[x1] - b[x1], a[x2] - b[x2]};
	return c;
}

Vec2D operator*(const Vec2D& a, const double d)
{
	Vec2D result = {a[x1]*d, a[x2]*d};
	return result;
}

Vec2D operator*(const double d, const Vec2D& a)
{
	Vec2D result = {a[x1]*d, a[x2]*d};
	return result;
}

Vec2D operator/(const Vec2D& a, const double d)
{
	assert(d != 0.0);
	//if(d == 0.0) cout << endl << "WARNING: division by zero scalar" << endl;
	Vec2D result = {a[x1]/d, a[x2]/d};
	return result;
}

double operator*(const Vec2D& a, const Vec2D& b)
{
	double dot = a[x1]*b[x1] + a[x2]*b[x2];
	return dot;
}

bool operator==(const Vec2D& a, const Vec2D& b)
{
	if(a[x1] == b[x1] && a[x2] == b[x2]) return true;
	else return false;
}

bool Vec2DCompare(const Vec2D& a, const Vec2D& b)
{
	if(a[x2] < b[x2])
	{
		return true;
	}
	else if(a[x2] == b[x2])
	{
		if(a[x1] < b[x1])
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	else
		return false;
}
/////////////////////////////////////////////////////////////////////////////////
//matrix operators
/////////////////////////////////////////////////////////////////////////////////
Mat2D operator+(const Mat2D& a, const Mat2D& b)
{
	Mat2D c = {a[a11] + b[a11], a[a12] + b[a12], a[a21] + b[a21], a[a22] + b[a22]};
	return c;
}

Mat2D operator-(const Mat2D& a, const Mat2D& b)
{
	Mat2D c = {a[a11] - b[a11], a[a12] - b[a12], a[a21] - b[a21], a[a22] - b[a22]};
	return c;
}

Mat2D operator*(const Mat2D& a, const Mat2D& b)
{
	Mat2D c = {0.0, 0.0, 0.0, 0.0};
	for(int i = 0; i < 2; i++)//row in c and in a
	{
		for(int j = 0; j < 2; j++)//column in c and in b
		{
			for(int k = 0; k < 2; k++)//column in a and row in b
			{
				int ind = i*2+j;
				int a_ind = i*2+k;
				int b_ind = k*2+j;
				c[ind] += a[a_ind]*b[b_ind];
			}
		}
	}
	return c;
}

Mat2D operator*(const Mat2D& a, const double d)
{
	Mat2D result = {a[a11]*d, a[a12]*d, a[a21]*d, a[a22]*d};
	return result;
}

Mat2D operator*(const double d, const Mat2D& a)
{
	Mat2D result = {a[a11]*d, a[a12]*d, a[a21]*d, a[a22]*d};
	return result;
}

void operator+=(Mat2D& a, const Mat2D& b)
{
	a = {a[a11] + b[a11], a[a12] + b[a12], a[a21] + b[a21], a[a22] + b[a22]};
}

Mat2D Mat2DInv(const Mat2D& a)
{
    double det = a[a11]*a[a22] - a[a12]*a[21];
    assert(det != 0.0);
    Mat2D result = {a[a22]/det,-a[a12]/det,-a[a21]/det,a[a11]/det};
    return result;
}

//vector-matrix multiplication
Vec2D operator*(const Mat2D& a, const Vec2D& b)
{
	Vec2D c = {0.0, 0.0};
	for(int i = 0; i < 2; i++)
	{
		for(int j = 0; j < 2; j++)
		{
			int ind = i*2+j;
			c[i] += a[ind]*b[j];
		}
	}
	return c;
}

//printing
ostream& operator<<(std::ostream& os, const Vec2D& v)
{
    unsigned int width = 15;
	std::ios state(NULL);
	state.copyfmt(os);
    os.precision(3);
    //os << scientific;
    //os.precision(3);
    os << fixed;
	
    os << "[" << setw(width) << v[x1]  << setw(width) << v[x2] << "]";
	
	os.copyfmt(state);
	return os;
}

ostream& operator<<(std::ostream& os, const Mat2D& m)
{
    unsigned int width = 15;
	std::ios state(NULL);
	state.copyfmt(os);
    os.precision(3);
    //os << scientific;
    //os.precision(3);
    os << fixed;
	
    os << "[" << setw(width) << m[a11] << setw(width) << m[a12] << "; " <<  setw(width) << m[a21] <<  setw(width) << m[a22] << "]";
	
	os.copyfmt(state);
	return os;
}
//updating vector values
void SetVec2D(Vec2D& v, const double a, const double b)
{
	v[x1] = a; v[x2] = b;
}
//vector length
double Vec2DLength(const Vec2D& v)
{
	return v[x1]*v[x1] + v[x2]*v[x2];
}

//sign function
int sign(double num)
{
	return (double(0) < num) - (num < double(0));
}
