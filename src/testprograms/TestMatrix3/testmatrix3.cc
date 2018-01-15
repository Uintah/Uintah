/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <testprograms/TestMatrix3/testmatrix3.h>
#include <Core/Math/Matrix3.h>

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <unistd.h>

void doMatrixSolvingTests(Suite& suite);
void doEigenTests(Suite& suite);
void doPolarDecompTests(Suite& suite);
void doEigenPlaneTests(Suite& suite);

void addSolveTests(Suite& suite, const std::string& test_name,
	       const Uintah::Matrix3& M, const Uintah::Vector& rhs, bool exp_return,
	       int exp_xg_basis_size);

//bool testEigenValue(const Uintah::Matrix3& M, double eigen_value);
bool testEigenValue(const Uintah::Matrix3& M, double eigen_value, double max_eigen_value);

void randomlyMixup(Uintah::Matrix3& M, Uintah::Vector& rhs);

bool equal_enough(double x1, double x2, double rel_scale);
bool equal_enough(Uintah::Vector v1, Uintah::Vector v2, double rel_scale);

double getRandom(); // returns a random number between -100.000 and 100.000

// returns 0 with the given probability or a random number
// between -100.000 and 100.000 otherwise.
double getRandomOrZero(float probability_for_zero); 

Uintah::Vector randomVector();
Uintah::Matrix3 randomMatrix();

#define NEAR_ZERO 1e-7

void displayEigen(Uintah::Matrix3 M);

#if 0

// This is a library... can't have a main() function....

int main()
{

  Suite test("m3test");

//  Uintah::Matrix3 testmat(1.0,1.5,0.0,
//                  0.0,1.0,0.0,
//                  0.0,0.0,1.0);

  for(int i=0;i<1000;i++){

    Uintah::Matrix3 testmat(drand48(),drand48(),drand48(),
                    drand48(),drand48(),drand48(),
                    drand48(),drand48(),drand48());

    cout << "testmat = " << std::endl;
    cout << testmat << std::endl;

    if(testmat.Determinant()<=0.0){
      cout << "Skipping this singular test matrix" << std::endl;
      cout << "Det(testmat) = " << testmat.Determinant() << std::endl;
    } else{

      Uintah::Matrix3 R, U;

      testmat.polarDecomposition(U,R,1e-10,true);

      cout << "U = " << std::endl;
      cout << U  << std::endl;

      cout << "R = " << std::endl;
      cout << R  << std::endl;

      cout << "R^T*R = " << std::endl;
      cout << R.Transpose()*R  << std::endl;

      cout << "R*U = " << std::endl;
      cout << R*U  << std::endl;


      testmat.polarDecompositionRMB(U,R);
//  testmat.polarRotationRMB(R);

      cout << "U = " << std::endl;
      cout << U  << std::endl;

      cout << "R = " << std::endl;
      cout << R  << std::endl;

      cout << "R^T*R = " << std::endl;
      cout << R.Transpose()*R  << std::endl;

      cout << "R*U = " << std::endl;
      cout << R*U  << std::endl;

      cout << "Success!" << std::endl;
   }
  }

  return 1;
}
#endif

SuiteTree* matrix3TestTree()
{
  srand(getpid());  
  SuiteTreeNode* matrix3Tests = new SuiteTreeNode("Uintah::Matrix3");
    
  Suite* solvingTests = new Suite("Solving Ax=b");
  Suite* eigenTests = new Suite("Eigen values/vectors");
  Suite* eigenPlaneTests = new Suite("Eigen plane values");
  Suite* polarDecompTests = new Suite("Polar Decomposition Tests");

  doMatrixSolvingTests(*solvingTests);
  doEigenTests(*eigenTests);
  doEigenPlaneTests(*eigenPlaneTests);
  doPolarDecompTests(*polarDecompTests);
  matrix3Tests->addSuite(solvingTests);
  matrix3Tests->addSuite(eigenTests);
  matrix3Tests->addSuite(eigenPlaneTests);
  matrix3Tests->addSuite(polarDecompTests);

  return matrix3Tests;

}

void displayEigen(Uintah::Matrix3 M)
{
  double e[3];
  int n = M.getEigenValues(e[0], e[1], e[2]);
  for (int i = 0; i < n; i++) {
    std::cout << "Eigen value: " << e[i] << std::endl;
    std::vector<Uintah::Vector> eigenVectors = M.getEigenVectors(e[i], M.MaxAbsElem());
    std::cout << "Eigen vectors:\n";
    for (int j = 0; j < (int)eigenVectors.size(); j++)
      std::cout << eigenVectors[j] << std::endl;
  }
  std::cout << std::endl;
}

void doMatrixSolvingTests(Suite& suite)
{
  Uintah::Vector rhs;
  Uintah::Vector xp;
  std::vector<Uintah::Vector> xg_basis;
  Uintah::Matrix3 M;

  int i = 0;
  std::string test_name_base = "Point: {{1, 0, 0} {0, 1, 0} {0, 0, 1}}";
  bool has_solution = true;
  double a, b, c, e;

  // run 10 random tests of matrix equations of one point solution
  for (i = 0; i < 10; i++) {
    M.Identity();
    rhs = randomVector();
    randomlyMixup(M, rhs); // this is what will really test it
    addSolveTests(suite, test_name_base, M, rhs, has_solution, 0);
  }
  
  // run 10 random tests for matrices that reduce to the following type
  test_name_base = "Line: {{1, 0, c} {0, 1, e} {0, 0, 0}}";
  xg_basis.resize(1);
  for (i = 0; i < 10; i++) {
    M.Identity();
    c = getRandom();
    e = getRandom();
    M(2, 2) = 0; M(0, 2) = c; M(1, 2) = e;
    rhs = randomVector();
    rhs.z(getRandomOrZero(0.75)); // make rhs.z zero 75% of the time
    has_solution = (rhs.z() == 0);

    randomlyMixup(M, rhs); // this is what will really test it
    addSolveTests(suite, test_name_base, M, rhs, has_solution, 1);
  }

   // run 10 random tests for matrices that reduce to the following type
  test_name_base = "Line: {{1, b, 0} {0, 0, 1} {0, 0, 0}}";
  for (i = 0; i < 10; i++) {
    M.Identity();
    b = getRandom();
    M(2, 2) = 0; M(1, 1) = 0; M(1, 2) = 1; M(0, 1) = b;
    rhs = randomVector();
    rhs.z(getRandomOrZero(0.75)); // make rhs.z zero 75% of the time
    has_solution = (rhs.z() == 0);

    randomlyMixup(M, rhs); // this is what will really test it
    addSolveTests(suite, test_name_base, M, rhs, has_solution, 1);
  }

  test_name_base = "Line: {{0, 1, 0} {0, 0, 1} {0, 0, 0}}";
  for (i = 0; i < 10; i++) {
    M.Identity();
    b = getRandom();
    M(0, 0) = 0; M(0, 1) = 1; M(1, 1) = 0; M(1, 2) = 1; M(2, 2) = 0;
    rhs = randomVector();
    rhs.z(getRandomOrZero(0.75)); // make rhs.z zero 75% of the time
    has_solution = (rhs.z() == 0);

    randomlyMixup(M, rhs); // this is what will really test it
    addSolveTests(suite, test_name_base, M, rhs, has_solution, 1);
  }
  test_name_base = "Plane: {{a, b, c} {0, 0, 0} {0, 0, 0}}";
  for (i = 0; i < 100; i++) {
    M.Identity();
    a = getRandomOrZero(0.5);
    b = getRandomOrZero(0.5);
    c = (a == 0 && b == 0) ? 1 : getRandomOrZero(0.5);
    M(0, 0) = a; M(0, 1) = b; M(0, 2) = c; M(1, 1) = 0; M(2, 2) = 0;
    rhs = randomVector();
    rhs.y(getRandomOrZero(0.80)); // make rhs.y zero 80% of the time
    rhs.z(getRandomOrZero(0.80)); // make rhs.z zero 80% of the time
    has_solution = ((rhs.z() == 0) && (rhs.y() == 0));

    randomlyMixup(M, rhs); // this is what will really test it
    addSolveTests(suite, test_name_base, M, rhs, has_solution, 2);
  }
  test_name_base = "All Space: {{0, 0, 0} {0, 0, 0} {0, 0, 0}}";
  M = Uintah::Matrix3();
  rhs = Uintah::Vector(0, 0, 0);
  addSolveTests(suite, test_name_base, M, rhs, true, 3);

  for (i = 0; i < 10; i++) {
    rhs = randomVector();
    addSolveTests(suite, test_name_base, M, rhs, (rhs == Uintah::Vector(0, 0, 0)), 3);
  }
}

void doEigenTests(Suite& suite)
{  
  Test* eigenValueOrderTest = suite.addTest("Eigenvalue e1, e2, e3 order");
  Test* threeEigenTestA = suite.addTest("Three eigenvalues/vectors, e1");
  Test* threeEigenTestB = suite.addTest("Three eigenvalues/vectors, e2");
  Test* threeEigenTestC = suite.addTest("Three eigenvalues/vectors, e3");
  Test* oneEigenTest = suite.addTest("One eigenvalue/vector");
  Uintah::Matrix3 M;
  double e1, e2, e3;
  int num_eigen_values;
  
  for (int i = 0; i < 100; i++) {
    // run a bunch of random tests
    M = randomMatrix();
    num_eigen_values = M.getEigenValues(e1, e2, e3);
    if (num_eigen_values == 1) {
      oneEigenTest->setResults(testEigenValue(M, e1, e1));
    }
    else if (num_eigen_values == 3) {
      threeEigenTestA->setResults(testEigenValue(M, e1, e1));
      threeEigenTestB->setResults(testEigenValue(M, e2, e1));
      threeEigenTestC->setResults(testEigenValue(M, e3, e1));
      eigenValueOrderTest->setResults((e1 > e2) && (e2 > e3));
    }
    else if (num_eigen_values == 2) {
      // two eigen values are the same
      // just treat this under the three eigen test case since
      // this is rare on random data
      threeEigenTestA->setResults(testEigenValue(M, e1, e1));
      threeEigenTestB->setResults(testEigenValue(M, e2, e1));
      eigenValueOrderTest->setResults(e1 > e2);    
    }
    else {
      suite.addTest("Bad number of eigen values", false);
    }
  }
  
}

void doPolarDecompTests(Suite& suite)
{  

  for(int i=0;i<1000;i++){

    Uintah::Matrix3 testmat(drand48(),drand48(),drand48(),
                    drand48(),drand48(),drand48(),
                    drand48(),drand48(),drand48());

    std::cout << "testmat = " << std::endl;
    std::cout << testmat << std::endl;

    if(testmat.Determinant()<=0.0){
      std::cout << "Skipping this singular test matrix" << std::endl;
      std::cout << "Det(testmat) = " << testmat.Determinant() << std::endl;
    } else{

      Uintah::Matrix3 R, U;

      testmat.polarDecomposition(U,R,1e-10,true);

      std::cout << "U = " << std::endl;
      std::cout << U  << std::endl;

      std::cout << "R = " << std::endl;
      std::cout << R  << std::endl;

      std::cout << "R^T*R = " << std::endl;
      std::cout << R.Transpose()*R  << std::endl;

      std::cout << "R*U = " << std::endl;
      std::cout << R*U  << std::endl;


      testmat.polarDecompositionRMB(U,R);
//  testmat.polarRotationRMB(R);

      std::cout << "U = " << std::endl;
      std::cout << U  << std::endl;

      std::cout << "R = " << std::endl;
      std::cout << R  << std::endl;

      std::cout << "R^T*R = " << std::endl;
      std::cout << R.Transpose()*R  << std::endl;

      std::cout << "R*U = " << std::endl;
      std::cout << R*U  << std::endl;

      std::cout << "Success!" << std::endl;
   }
  }



}

void doEigenPlaneTests(Suite& suite)
{  
  Test* eigenValueOrderTest = suite.addTest("e1 > e2");
  Test* eigenTestA[3];
  Test* eigenTestB[3];

  eigenTestA[0] = suite.addTest("YZ e1");
  eigenTestB[0] = suite.addTest("YZ e2");
  eigenTestA[1] = suite.addTest("XZ e1");
  eigenTestB[1] = suite.addTest("XZ e2");
  eigenTestA[2] = suite.addTest("XY e1");
  eigenTestB[2] = suite.addTest("XY e2");
  
  Uintah::Matrix3 M;
  Uintah::Matrix3 planeM; // 3d version of sub-matrix (with zeroes at bottom and right)
  double e1, e2;
  int num_eigen_values;
  
  for (int i = 0; i < 5; i++) {
    // run a bunch of random tests
    M = randomMatrix();
    for (int plane = 1; plane <= 3; plane++) {
      if (plane == 1) {
	num_eigen_values = M.getYZEigenValues(e1, e2);
	planeM = Uintah::Matrix3(M(1, 1), M(1, 2), 0, M(2, 1), M(2, 2), 0, 0, 0, 0);
      }
      else if (plane == 2) {
	num_eigen_values = M.getXZEigenValues(e1, e2);
	planeM = Uintah::Matrix3(M(0, 0), M(0, 2), 0, M(2, 0), M(2, 2), 0, 0, 0, 0);
      }
      else {
 	num_eigen_values = M.getXYEigenValues(e1, e2);
	planeM = Uintah::Matrix3(M(0, 0), M(0, 1), 0, M(1, 0), M(1, 1), 0, 0, 0, 0);
      }

      // Use the 3x3 test with planeM (which is the sub-matrix
      // with 0's on the bottom and right).  This works because
      // this every eigenvalue of the 2x2 sub-matrix should be
      // an eigenvalue of planeM.
      
      if (num_eigen_values == 1) {
	// two eigen values are the same
	eigenTestA[plane-1]->setResults(testEigenValue(planeM, e1, e1));
      }
      else if (num_eigen_values == 2) {
	eigenTestA[plane-1]->setResults(testEigenValue(planeM, e1, e1));
	eigenTestB[plane-1]->setResults(testEigenValue(planeM, e2, e1));
	eigenValueOrderTest->setResults(e2 < e1);    
      }
      else if (num_eigen_values != 0) {
	suite.addTest("Bad number of eigen values", false);
      }
    }
  }

}

bool testEigenValue(const Uintah::Matrix3& M, double eigen_value, double max_eigen_value)
{
  double rel_scale = fabs(max_eigen_value); //M.MaxAbsElem(); //M.Norm();
  std::vector<Uintah::Vector> eigenVectors = M.getEigenVectors(eigen_value, rel_scale);

  if (eigenVectors.size() == 0) {
    return false; // should have at least one Vector for an eigen_value
  }
  else if (eigenVectors.size() == 1) {
    // only a line of eigenVectors -- just test the one vector
    return equal_enough(M*eigenVectors[0], eigenVectors[0]*eigen_value,
			rel_scale);
  }
  else {
    // try different random linear combinations of eigenVectors
    bool success = true;
    for (int trials = 0; trials < 10; trials++) {
      Uintah::Vector x(0, 0, 0);
      for (int i = 0; i < (int)eigenVectors.size(); i++) {
	x = x + eigenVectors[i] * getRandom();
      }
      if (!equal_enough(M*x, x*eigen_value, rel_scale))
	success = false;
    }
    return success;
  }
}

void addSolveTests(Suite& suite, const std::string& test_name, const Uintah::Matrix3& M,
		   const Uintah::Vector& rhs, bool exp_return, int exp_xg_basis_size)
{
  Uintah::Vector xp;
  std::vector<Uintah::Vector> xg_basis;
  double rel_scale = M.MaxAbsElem();
  bool result = M.solve(rhs, xp, xg_basis, rel_scale);

  suite.findOrAddTest(test_name + ", existence", exp_return == result);

  if (result == true) {
    // this other stuff is only relevent if result == true
    suite.findOrAddTest(test_name + ", xp", equal_enough(M * xp, rhs,
							 rel_scale));
    /*    if (!equal_enough(M * xp, rhs, rel_scale)) {
      std::cout << rel_scale << std::endl;
      std::cout << xp << " != " << rhs << std::endl;
    }*/
	

    if (exp_xg_basis_size != (int)xg_basis.size()) {
      suite.findOrAddTest(test_name + ", xg size", false);
      suite.findOrAddTest(test_name + ", xg_basis", false);
    }
    else {
      // test the basis by making arbitrary linear combinations
      // of basis vectors and testing them out
      bool success = true;
      for (int trial = 0; trial < 50; trial++) {
	Uintah::Vector x(0, 0, 0);
	for (int i = 0; i < (int)xg_basis.size(); i++) {
	  x = x + xg_basis[i] * getRandom();
	}
	if (!equal_enough(M*x, Uintah::Vector(0, 0, 0), rel_scale))
	  success = false;
      }
      suite.findOrAddTest(test_name + ", xg_basis", success);
    }
  }
}

// returns a random number between -100.000 to 100.000
double getRandom()
{
  return (rand() % 200000 - 100000)/1000.0;
}

double getRandomOrZero(float probability_for_zero)
{
  if (rand() < probability_for_zero * RAND_MAX)
    return 0;
  else
    return getRandom();
}


// randomly mix up a matrix
void randomlyMixup(Uintah::Matrix3& M, Uintah::Vector& rhs)
{
  Uintah::Matrix3 new_M(M);
  Uintah::Vector new_rhs(rhs);
  int i, j, k;
  
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      double mult = getRandom();
      if (i != j) {
	for (k = 0; k < 3; k++)
	  new_M(i, k) += M(j, k) * mult;
	new_rhs[i] += rhs[j] * mult;
      }
      else {
	if (fabs(mult) > 1) {
	  for (k = 0; k < 3; k++)
	    new_M(i, k) *= mult;
	  new_rhs[i] *= mult;
	}
      }
    }
  }

  // now randomly swap rows
  int row_orders[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0},
			  {2, 0, 1}, {2, 1, 0}};
  int* row_order = row_orders[rand() % 6];
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      M(i, j) = new_M(row_order[i], j);
    }
    rhs[i] = new_rhs[row_order[i]];
//    rhs[i] = new_rhs[row_order[i]-1];
  }
}

Uintah::Vector randomVector()
{
  return Uintah::Vector(getRandom(), getRandom(), getRandom());
}

Uintah::Matrix3 randomMatrix()
{
  Uintah::Matrix3 M;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      M(i, j) = getRandom();
  return M * pow(10.0, rand() % 10 - 3);
}

bool equal_enough(double x1, double x2, double rel_scale)
{
  return fabs(x2 - x1) <= NEAR_ZERO * (rel_scale > 1 ? rel_scale : 1);
}

bool equal_enough(Uintah::Vector v1, Uintah::Vector v2, double rel_scale)
{
  for (int i = 0; i < 3; i++) {
    if (!equal_enough(v1[i], v2[i], rel_scale))
      return false;
  }
  return true;
}
