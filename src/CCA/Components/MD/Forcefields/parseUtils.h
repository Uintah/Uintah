/*
 * parseUtils.h
 *
 *  Created on: Mar 17, 2014
 *      Author: jbhooper
 */

#ifndef PARSEUTILS_H_
#define PARSEUTILS_H_

#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

namespace Parse {

  inline double stringToDouble(const std::string& string) {
    return std::strtod(string.c_str(), NULL);
  }

  inline int stringToInt(const std::string& string) {
    return std::strtol(string.c_str(), NULL, 10);
  }

  inline void tokenize(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ")
  {
      // Skip delimiters at beginning.
      std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
      // Find next delimiter after current string (which begins at lastPos).
      std::string::size_type pos     = str.find_first_of(delimiters, lastPos);

      while (std::string::npos != pos || std::string::npos != lastPos)
      {
          // Found a token, add it to the vector.
          tokens.push_back(str.substr(lastPos, pos - lastPos));
          // Skip delimiters.  Note the "not_of"
          lastPos = str.find_first_not_of(delimiters, pos);
          // Find next "non-delimiter"
          pos = str.find_first_of(delimiters, lastPos);
      }
  }

  inline void tokenizeAtMost(const std::string& str,
                             std::vector<std::string>& tokens,
                             const size_t numTokens = 0,
                             const std::string& delimiters = " \t")
  {
    // We parse tokens from left to right
    std::string::size_type left = 0;
    std::string::size_type right = 0;

    size_t tokenCount = 0;
    while ((std::string::npos != left || std::string::npos != right) && (tokenCount < numTokens)) {
      left  = str.find_first_not_of(delimiters,right);  // leftmost marker of string at first non-delimiter character
      right = str.find_first_of(delimiters,left);       // rightmost marker at first following delimiter
      tokens.push_back(str.substr(left, right - left)); // extract the substr between the two
      ++ tokenCount;                                    // and count it
    }

    left = str.find_first_not_of(delimiters,right);
    if (std::string::npos != left) { // Not at the end of the input line
      right = std::string::npos;
      if (left != right) tokens.push_back(str.substr(left, right-left));
    }
  }
}

#endif /* PARSEUTILS_H_ */
