#pragma once

#include"Typedefs.h"
#include<array>
#include<iostream>
#include<iomanip>
#include<cassert>
#include<cmath>

using namespace std;
//constants
const double TOL = 1e-15;
//how many particles is "ideal" per element
const unsigned int IDEAL_NP = 3;
//enums
enum class Output{NONE, TO_SCREEN, TO_FILE};
enum class MeshAction{Refine, Keep, Coarsen};
enum vec_entries{x1=0, x2=1};
enum mat_entries{a11=0, a12=1, a21=2, a22=3};
//node positions in the element: going counterclockwise:
//bottom-left, bottom-right, top-right, top-left
enum node_pos{_BL=0, _BR=1, _TR=2, _TL=3};
//vector operators
Vec2D operator+(const Vec2D& a, const Vec2D& b);
void operator+=(Vec2D& a, const Vec2D& b);
Vec2D operator-(const Vec2D& a, const Vec2D& b);
Vec2D operator*(const Vec2D& a, const double d);
Vec2D operator*(const double d, const Vec2D& a);
Vec2D operator/(const Vec2D& a, const double d);
double operator*(const Vec2D& a, const Vec2D& b);
bool operator==(const Vec2D& a, const Vec2D& b);
bool Vec2DCompare(const Vec2D& a, const Vec2D& b);
//matrix operators
Mat2D operator+(const Mat2D& a, const Mat2D& b);
Mat2D operator-(const Mat2D& a, const Mat2D& b);
Mat2D operator*(const Mat2D& a, const Mat2D& b);
Mat2D operator*(const Mat2D& a, const double d);
Mat2D operator*(const double d, const Mat2D& a);
void operator+=(Mat2D& a, const Mat2D& b);
Mat2D Mat2DInv(const Mat2D& a);
//vector-matrix multiplication
Vec2D operator*(const Mat2D& a, const Vec2D& b);
//setting values of vector
void SetVec2D(Vec2D& v, const double a, const double b);
//vector length
double Vec2DLength(const Vec2D& v);
//printing
ostream& operator<<(ostream& os, const Vec2D& v);
ostream& operator<<(ostream& os, const Mat2D& v);
//sign function
int sign(double num);
