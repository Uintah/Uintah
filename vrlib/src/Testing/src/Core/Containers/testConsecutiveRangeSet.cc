#include <Core/Containers/ConsecutiveRangeSet.h>
#include <iostream>

int main() {
  const std::string className = "ConsecutiveRangeSet";

  std::cerr << "Testing constructors\n";

  // test constructor ConsecutiveRangeSet(std::list<int>& set);
  std::list<int> list;
  list.push_back(1);
  list.push_back(3);
  list.push_back(5);
  SCIRun::ConsecutiveRangeSet set1( list );
  if (set1.expandedString() != "1 3 5") {
    std::cerr << className << ": set1 expandedString value incorrect.\n";
    return -1;
  }
  
  // test constructor ConsecutiveRangeSet(int low, int high);
  SCIRun::ConsecutiveRangeSet set2(3, 7);
  if (set2.expandedString() != "3 4 5 6 7") {
    std::cerr << className << ": set2 expandedString value incorrect.\n";
    return -1;
  }

  // test constructor ConsecutiveRangeSet()
  SCIRun::ConsecutiveRangeSet set3;
  if (set3.expandedString() != "") {
    std::cerr << className << ": set3 expandedString value incorrect.\n";
    return -1;
  }  

  // test constructor ConsecutiveRangeSet(const std::string& setstr)
  SCIRun::ConsecutiveRangeSet set4("1, 3-5, 7, 9-11");
  if (set4.expandedString() != "1 3 4 5 7 9 10 11") {
    std::cerr << className << ": set4 expandedString value incorrect.\n";
    return -1;
  }

  // test constructor ConsecutiveRangeSet(const ConsecutiveRangeSet& set2)
  SCIRun::ConsecutiveRangeSet set5( set4 );
  if (set5.expandedString() != "1 3 4 5 7 9 10 11") {
    std::cerr << className << ": set5 expandedString value incorrect.\n";
    return -1;
  }

  std::cerr << "Done testing constructors\n"; 

  // test operator=
  SCIRun::ConsecutiveRangeSet set6 = set5; 
  if (set6.expandedString() != "1 3 4 5 7 9 10 11") {
    std::cerr << className << ": set6 expandedString value incorrect. Overloaded operator= possible problem.\n";
    return -1;
  } 

  // test addInOrder
  std::cerr << "Testing addInOrder\n";
  set3.addInOrder(1);
  set3.addInOrder(3);
  set3.addInOrder(5);
  if (set3.expandedString() != "1 3 5") {
    std::cerr << className << ": addInOrder calls produced incorrect results.\n";
    return -1;
  }

  // test operator==
  std::cerr << "Testing operator==\n";
  if (set1 == set2) {
    std::cerr << className << ": operator== produced incorrect results.\n";
    return -1;
  }

  // test operator!=
  std::cerr << "Testing operator!=\n";
  if (set1 != set3) {
    std::cerr << className << ": operator!= produced incorrect results.\n";
    return -1;
  }

  // test iterators
  std::cerr << "Testing iterators\n";
  SCIRun::ConsecutiveRangeSet::iterator iter = set1.begin();
  int count=0;
  while(iter != set1.end()) {
    std::cerr << "Iterating: " << *iter << std::endl;
    count++;
    iter++;
  }
  if (count != 3) {
    std::cerr << className << ": did not correctly iterate over set1\n";
    return -1;
  }

  // test intersected
  std::cerr << "Testing intersected function\n";
  SCIRun::ConsecutiveRangeSet set7 = set1.intersected(set2);
  if (set7.expandedString() != "3 5") {
    std::cerr << className << ": intersected call produced incorrect results.\n";
    return -1;
  }  

  // test unioned
  std::cerr << "Testing unioned function\n";
  SCIRun::ConsecutiveRangeSet set8("2, 4, 6");
  SCIRun::ConsecutiveRangeSet set9 = set1.unioned(set8);
  if (set9.expandedString() != "1 2 3 4 5 6") {
    std::cerr << className << ": unioned call produced incorrect results.\n";
    return -1;
  }  
  
  return 0;
}
