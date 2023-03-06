/*!\file
* \brief adding span type to namespace std in pre-adaptation for C++20
*
* https://en.cppreference.com/w/cpp/container/span
*/
/*
 * span.hpp
 *
 *      Author: David Lawrie
 */

#ifndef SPAN_HPP_
#define SPAN_HPP_

#include "../3P/_outside_libraries/span-lite/span.hpp"

namespace std{

///adding span type to namespace std in pre-adaptation for C++20
using nonstd::dynamic_extent;
using nonstd::span;
using nonstd::with_container;
using nonstd::operator==;
using nonstd::operator!=;
using nonstd::operator<;
using nonstd::operator<=;
using nonstd::operator>;
using nonstd::operator>=;
//using nonstd::span_lite::make_span;

} /* ----- end namespace std ----- */

#endif /* SPAN_HPP_ */
