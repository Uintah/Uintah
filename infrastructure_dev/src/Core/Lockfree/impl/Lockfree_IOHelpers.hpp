#ifndef LOCKFREE_IMPL_IO_HELPERS_HPP
#define LOCKFREE_IMPL_IO_HELPERS_HPP

#include <string>
#include <sstream>
#include <iomanip>

namespace Lockfree { namespace Impl {

inline std::string bytes_to_string( uint64_t bytes )
{
  enum {
     KB = 1ull << 10
   , MB = 1ull << 20
   , GB = 1ull << 30
   , TB = 1uLL << 40
   , PB = 1ull << 50
  };

  std::ostringstream out;

  if ( bytes < KB ) {
    out << bytes << " B";
  }
  else if ( bytes < MB ) {
    out << std::setprecision(5) << (static_cast<double>(bytes) / KB) << " KB";
  }
  else if ( bytes < GB ) {
    out << std::setprecision(5) << (static_cast<double>(bytes) / MB) << " MB";
  }
  else if ( bytes < TB ) {
    out << std::setprecision(5) << (static_cast<double>(bytes) / GB) << " GB";
  }
  else if ( bytes < PB ) {
    out << std::setprecision(5) << (static_cast<double>(bytes) / TB) << " TB";
  }
  else {
    out << std::setprecision(5) << (static_cast<double>(bytes) / PB) << " PB";
  }

  return out.str();
}


inline std::string big_round_number( double i )
{
  enum : uint64_t {
     HUNDRED = 100ull
   , THOUSAND = 10ull * HUNDRED
   , MILLION  = THOUSAND * THOUSAND
   , BILLION  = THOUSAND * MILLION
   , TRILLION = THOUSAND * BILLION
  };

  std::ostringstream out;

  if ( i < HUNDRED ) {
    out << std::setprecision(4) << i;
  }
  else if ( i < THOUSAND ) {
    out << std::setprecision(4) << (i/HUNDRED) << " hundred";
  }
  else if ( i < MILLION ) {
    out << std::setprecision(4) << (i/THOUSAND) << " thousand";
  }
  else if ( i < BILLION ) {
    out << std::setprecision(4) << (i/MILLION) << " million";
  }
  else if ( i < TRILLION ) {
    out << std::setprecision(4) << (i/BILLION) << " billion";
  }
  else {
    out << std::setprecision(4) << (i/TRILLION) << " trillion";
  }

  return out.str();
};


}} // namespace Lockfree::Impl


#endif // LOCKFREE_IMPL_IO_HELPERS_HPP
