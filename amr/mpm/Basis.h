#pragma once

#include<iostream>
#include<string>
#include<cmath>
#include<stdlib.h>
#include"Utils.h"
#include"BoundingBox.h"
using namespace std;
//class representing hat function restricted to the given element
class Basis
{
	public:
		Basis(const Vec2D& _org, const double _dx, const double _dy):
			org(_org), dx(_dx), dy(_dy){};
		//printout basis information
		void Print(const string& msg);
		double Eval(const Vec2D& v);
		//for gradient evaluation
		double dxEval(const Vec2D& v);
		double dyEval(const Vec2D& v);
		Vec2D Grad(const Vec2D& v);
	private:
		//BoundingBox bBox;
		Vec2D org;
		double dx;
		double dy;
};
