/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <CCA/Components/MPM/PhysicalBC/PolynomialData.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/Point.h>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace Uintah;
using namespace std;


PolynomialData::PolynomialData(ProblemSpecP& ps, const Point& bottom, 
                               const Point& top)
{
  ProblemSpecP polyData = ps->findBlock("polynomial_data");
  if (!polyData)
    throw ProblemSetupException("**ERROR** No polynomial data specified.",
                                __FILE__,__LINE__);

  d_endCapName = "";

  for (ProblemSpecP fileData = polyData->findBlock("file"); fileData != 0;
       fileData = fileData->findNextBlock("file")) {
    
    string fileName;
    fileData->get(fileName);

    if (fileName.find("endcap") != string::npos) {
      d_endCapName = fileName;
    }
    else
      d_fileNames.push_back(fileName);
  }

  d_bottom = bottom;
  d_top = top;
              
  loadData();

}

PolynomialData::~PolynomialData()
{
}

void PolynomialData::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP pd_ps = ps->appendChild("polynomial_data");
  for (vector<std::string>::const_iterator itr = d_fileNames.begin(); 
       itr != d_fileNames.end(); itr++) {
    pd_ps->appendElement("file",*itr);
  }
  if (d_endCapName != "")
    pd_ps->appendElement("file",d_endCapName);

}

void PolynomialData::loadData()
{

  for (unsigned int i = 0; i < d_fileNames.size(); i++) {

    ifstream polyFile(d_fileNames[i].c_str());
    if (!polyFile) {
      ostringstream warn;
      warn << "ERROR: opening polynomial data file: "<< d_fileNames[i].c_str();
      throw ProblemSetupException(warn.str(),__FILE__, __LINE__);
    }

    double data;
    vector<double> dataList;
   
    // Need to ignore the first  line of data.
    
    // Line 1: axial location
    // Line 2: A
    // Line 3: B
    // Line 4: C
    // Line 5: D
    // Line 6: E

    // Heat Flux = A + B*sin(theta) + C*cos(theta) + D*sin(2*theta) +
    //             E*cos(2*theta)

    polyFile >> data;

    while (polyFile >> data) {
      //cout << "data = " << data << endl;
      dataList.push_back(data);
    }
    d_polyData.push_back(dataList);
    polyFile.close();
  }

  if (d_endCapName != "") {
    ifstream endCapFile(d_endCapName.c_str());
    if (!endCapFile) {
      ostringstream warn;
      warn << "ERROR: opening polynomial data file: "<< d_endCapName.c_str();
      throw ProblemSetupException(warn.str(),__FILE__, __LINE__);
    }
    endCapFile >> d_endCapLow;
    endCapFile >> d_endCapHigh;

    //cout << "d_endCapLow = " << d_endCapLow << " d_endCapHigh = " << d_endCapHigh << endl;

    vector<double> dataList;
    dataList.push_back(d_endCapLow);
    for (int i = 1; i < 5; i++)
      dataList.push_back(0.);

    d_polyData.push_front(dataList);

    dataList.clear();

    dataList.push_back(d_endCapHigh);
    for (int i = 1; i < 5; i++)
      dataList.push_back(0.);

    d_polyData.push_back(dataList);
  }

  Vector diff = d_top.asVector() - d_bottom.asVector();
  double increment = diff.length()/(d_polyData.size() - 1);
  
  //cout << "size of d_polyData = " << d_polyData.size() << endl;
  //cout << "size of d_fileNames = " << d_fileNames.size() << endl;

  for (unsigned int j = 0; j < d_polyData.size(); j++) {
    double t =  increment*static_cast<double>(j);
    d_polyRange.push_back(t);
  }
  
}

double PolynomialData::interpolateRadial(const int polyNum, const double theta)
{

  double value = 0.;

  value += d_polyData[polyNum][0];
  value += d_polyData[polyNum][1]* sin(theta);
  value += d_polyData[polyNum][2]* cos(theta);
  value += d_polyData[polyNum][3]* sin(2.*theta);
  value += d_polyData[polyNum][4]* cos(2.*theta);

  value = max(0.0, value);       // clamp
  return value;
}


double PolynomialData::interpolateValue(const Point& test_pt)
{

  Vector axis = d_top - d_bottom;
  Vector pt_bottom = test_pt.asVector() - d_bottom;

  // Project the vector made up of the test_pt to the axis
  // and determine its length to determine which set of polynomials need to
  // use for linear interpolation along the axis of the cylinder.
  double scaledVecLength = Dot(axis,pt_bottom)/(axis.length());

  // find the min and max ids for the polynomial range based on the 
  // scaledVecLength.

  int min = 0;
  int max = 0;
  for (unsigned int i = 0; i < d_polyRange.size(); i++) {
    if (d_polyRange[i] <=  scaledVecLength && 
        scaledVecLength <= d_polyRange[i+1] ) {
      min = i;
      max = i+1;
    }
    if (d_polyRange[i] == scaledVecLength) {
      min = i;
      max = i;
    }

  }
  
  double theta = atan(test_pt.y()/test_pt.x());
  // First quadrant
  if (test_pt.x() >= 0 && test_pt.y() >= 0) {
    theta += 2.*M_PI;
  }

  // Second quadrant
  if (test_pt.x() < 0 && test_pt.y() >= 0) {
    theta += M_PI;
  }

  // Third quadrant
  if (test_pt.x() < 0 && test_pt.y() < 0) {
    theta += M_PI;

  }

  // Fourth quadrant
  if (test_pt.x() >= 0 && test_pt.y() < 0) {
    theta += 2.* M_PI;

  }

  double minValue = interpolateRadial(min,theta);
  double maxValue = interpolateRadial(max,theta);

#if 1
  if (minValue < 0. || maxValue < 0.)
    cout << "WARNING values less than 0" << endl;
#endif


  // y = m*x + b for doing linear interpolation between two polynomial curves.
  double m = (maxValue - minValue)/(d_polyRange[max] - d_polyRange[min]);

  double b = minValue - m*d_polyRange[min];
  
  double value = m * scaledVecLength + b;

  if (value > minValue && value > maxValue)
    cout << "WARNING values computed incorrectly" << endl;

  if (value < minValue && value < maxValue)
    cout << "WARNING values computed incorrectly" << endl;
    

  return value;
  
}
