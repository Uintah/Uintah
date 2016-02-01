#ifndef LOCKFREE_USAGE_MODEL_HPP
#define LOCKFREE_USAGE_MODEL_HPP

#include "impl/Lockfree_Macros.hpp"       // for LOCKFREE_ENABLE_CXX11

namespace Lockfree {

/// enum UsageModel
///
/// SHARED_INSTANCE -- many threads can shared an instance
/// EXCLUSIVE_INSTANCE -- one and only one thread will access an instance
enum UsageModel { SHARED_INSTANCE, EXCLUSIVE_INSTANCE };

enum SizeModel { ENABLE_SIZE = true, DISABLE_SIZE = false };

} // namespace Lockfree


#endif //LOCKFREE_USAGE_MODEL_HPP
