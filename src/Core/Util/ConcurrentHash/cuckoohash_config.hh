/* This file originally came from the the libcuckoo project: https://github.com/efficient/libcuckoo
 * Authors: Manu Goyal, Bin Fan, Xiaozhou Li, David G. Andersen, and Michael Kaminsky
 * It was released under the Apache 2.0 license:
 *
 * Copyright (C) 2013, Carnegie Mellon University and Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
 * except in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the
 * License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 *
 *
 * The Apache license requires that we detail all changes made to the original file.  Those changes are listed below:
 *
*/
/** \file */

#ifndef _CUCKOOHASH_CONFIG_HH
#define _CUCKOOHASH_CONFIG_HH

#include <cstddef>
#include <limits>

//! The default maximum number of keys per bucket
constexpr size_t LIBCUCKOO_DEFAULT_SLOT_PER_BUCKET = 4;

//! The default number of elements in an empty hash table
constexpr size_t LIBCUCKOO_DEFAULT_SIZE =
    (1U << 16) * LIBCUCKOO_DEFAULT_SLOT_PER_BUCKET;

//! The default minimum load factor that the table allows for automatic
//! expansion. It must be a number between 0.0 and 1.0. The table will throw
//! libcuckoo_load_factor_too_low if the load factor falls below this value
//! during an automatic expansion.
constexpr double LIBCUCKOO_DEFAULT_MINIMUM_LOAD_FACTOR = 0.05;

//! An alias for the value that sets no limit on the maximum hashpower. If this
//! value is set as the maximum hashpower limit, there will be no limit. This
//! is also the default initial value for the maximum hashpower in a table.
constexpr size_t LIBCUCKOO_NO_MAXIMUM_HASHPOWER =
    std::numeric_limits<size_t>::max();

//! set LIBCUCKOO_DEBUG to 1 to enable debug output
#define LIBCUCKOO_DEBUG 0

#endif // _CUCKOOHASH_CONFIG_HH
