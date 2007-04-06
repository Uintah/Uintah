#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/PolynomialData.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/Point.h>
#include <iostream>
#include <fstream>


using namespace Uintah;
using namespace std;


PolynomialData::PolynomialData(ProblemSpecP& ps, const Point& bottom, 
                               const Point& top)
{
  ProblemSpecP polyData = ps->findBlock("polynomial_data");
  if (!polyData)
    throw ProblemSetupException("**ERROR** No polynomial data specified.",
                                __FILE__,__LINE__);

  for (ProblemSpecP fileData = polyData->findBlock("file"); fileData != 0;
       fileData = fileData->findNextBlock("file")) {
    
    string fileName;
    fileData->get(fileName);
    d_fileNames.push_back(fileName);
  }

  d_bottom = bottom;
  d_top = top;
              
  //cout << "number of files read " << d_fileNames.size() << endl;
  loadData();

}

PolynomialData::~PolynomialData()
{

}

void PolynomialData::loadData()
{

  for (unsigned int i = 0; i < d_fileNames.size(); i++) {
    // cout << "opening file: " << d_fileNames[i] << endl;

    ifstream polyFile(d_fileNames[i].c_str());
    if (!polyFile) {
          throw ProblemSetupException("ERROR: opening polynomial data file",
                                      __FILE__, __LINE__);
    }

    double data;
    vector<double> dataList;
   
    // Need to ignore the first 3 lines of data.
    
    // Line 1: axial location
    // Line 2: start theta (you can ignore)
    // Line 3: end theta (again ignore)
    // Line 4: A
    // Line 5: B
    // Line 6: C

    // Heat Flux = A + B*sin(theta) + C*cos(theta) + D*sin(2*theta) +
    //             E*cos(2*theta)

    polyFile >> data >> data >> data;

    while (polyFile >> data) {
      //cout << "data = " << data << endl;
      dataList.push_back(data);
    }
    d_polyData.push_back(dataList);
    polyFile.close();
  }

  Vector diff = d_top.asVector() - d_bottom.asVector();
  double increment = diff.length()/(d_fileNames.size() - 1);
  
  for (unsigned int j = 0; j < d_fileNames.size(); j++) {
    double t =  increment*static_cast<double>(j);
    d_polyRange.push_back(t);
  }
  
}

double PolynomialData::interpolateRadial(const int polyNum, const double theta)
{

  double value = 0.;
  int order = 0;
  value += d_polyData[polyNum][0];
  value += d_polyData[polyNum][1]* cos(theta);
  value += d_polyData[polyNum][2]* sin(theta);
  value += d_polyData[polyNum][3]* cos(2.*theta);
  value += d_polyData[polyNum][4]* sin(2.*theta);

#if 0
  for (vector<double>::iterator iter = d_polyData[polyNum].begin(); 
       iter != d_polyData[polyNum].end(); iter++) {
    //value += *iter * pow(theta,order);
    value += *iter * cos(order*theta)
    cout << "theta = " << theta << " polynomial = " << *iter << " value = " << value << endl;
    order++;
  }
#endif

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

  //  cout << "minValue = " << minValue << " maxValue = " << maxValue << endl;
  if (minValue < 0. || maxValue < 0.)
    cout << "WARNING values less than 0" << endl;


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
