#include <Core/Containers/StringUtil.h>
#include <iostream>

int main() {
  const std::string int_string = "123";
  int int_result = 0;
  if (!SCIRun::string_to_int(int_string, int_result)) {
    std::cerr << "StringUtil string_to_int failed\n";
    return -1;
  } else if (int_result != 123) {
    std::cerr << "StringUtil string_to_int returned ";
    std::cerr << "incorrect results: " << int_result << "\n";
    return -1;
  }

  const std::string double_string = "1.123456789";
  double double_result = 0;
  if (!SCIRun::string_to_double(double_string, double_result)) {
    std::cerr << "StringUtil string_to_double failed\n";
    return -1;
  } else if (double_result != 1.123456789) {
    std::cerr << "StringUtil string_to_double returned ";
    std::cerr << "incorrect results: " << double_result << "\n";
    return -1;
  }

  const std::string unsigned_long_string = "429496729";
  unsigned long unsigned_long_result = 0;
  if (!SCIRun::string_to_unsigned_long(unsigned_long_string, 
				       unsigned_long_result)) {
    std::cerr << "StringUtil string_to_unsigned_long failed\n";
    return -1;
  } else if (unsigned_long_result != 429496729) {
    std::cerr << "StringUtil string_to_unsigned_long returned ";
    std::cerr << "incorrect results: " << unsigned_long_result << "\n";
    return -1;
  }

  
  return 0;
}
