#ifndef LOCKFREE_MACROS_HPP
#define LOCKFREE_MACROS_HPP

// supported compilers
// gcc >= 4.8
// clang >= 3.5
// intel >= 15

// include sized integer types
#include <cstdint>
#include <cstddef>

// get standard limit macros
#include <climits>
#include <limits>

// std::move and std::forward
#include <utility>

#include <type_traits>

// wrap compiler attributes
#define LOCKFREE_FORCEINLINE inline __attribute__((always_inline))
#define LOCKFREE_MAY_ALIAS __attribute__((__may_alias__))


// detect support for alignas and alignof
#if defined(__clang__)
	/* Clang/LLVM. ---------------------------------------------- */
  #define LOCKFREE_NOEXCEPT noexcept
  #if (__clang_major__ > 3) || ((__clang_major__ == 3) && (__clang_minor__ > 2))
    #define LOCKFREE_ALIGNOF(T) alignof(T)
    #define LOCKFREE_ALIGNAS(T) alignas(T)
  #else
    #define LOCKFREE_ALIGNOF(T) 0
    #define LOCKFREE_ALIGNAS(T)
  #endif
#elif defined(__ICC) || defined(__INTEL_COMPILER)
	/* Intel ICC/ICPC. ------------------------------------------ */
  #define LOCKFREE_NOEXCEPT
  #if (__INTEL_COMPILER >= 1500)
    #define LOCKFREE_ALIGNOF(T) alignof(T)
    #define LOCKFREE_ALIGNAS(T) alignas(T)
  #else
    #define LOCKFREE_ALIGNOF(T) 0
    #define LOCKFREE_ALIGNAS(T)
  #endif
#elif defined(__GNUC__) || defined(__GNUG__)
	/* GNU GCC/G++. --------------------------------------------- */
  #define LOCKFREE_NOEXCEPT noexcept
  #if (__GNUC__ > 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 8))
    #define LOCKFREE_ALIGNOF(T) alignof(T)
    #define LOCKFREE_ALIGNAS(T) alignas(T)
  #else
    #define LOCKFREE_ALIGNOF(T) 0
    #define LOCKFREE_ALIGNAS(T)
  #endif
#endif

#endif //LOCKFREE_MACROS_HPP
