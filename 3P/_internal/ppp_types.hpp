/*!\file
* \brief simple range and span types for easy iterator/pointer range access
*
*/
/*
 * ppp_types.hpp
 *
 *      Author: David Lawrie
 */

#ifndef PPP_TYPES_HPP_
#define PPP_TYPES_HPP_

#include <algorithm>
#include "../span.hpp"

namespace ppp{

/* ----- unique_span wrapper ----- */
template<typename T>
struct delete_host_span{
	inline void operator()(T * s) noexcept{ if(s != nullptr) { delete []s; s = nullptr; } }
};

template<typename T, template<typename> class span_Type>
struct make_host_span{
	inline span_Type<T> operator()(size_t num_elements) {
		T* ptr = new T[num_elements];
		return span_Type<T>(ptr,num_elements);
	}
};

template<typename T, template<typename> class span_Type, template<typename> class constructor, template<typename> class deleter>
struct unique_span{
	inline unique_span() noexcept = default;
	inline unique_span(size_t num_elements): make_span{}, destroy_span{}, my_span{make_span(num_elements)}{ }
	inline unique_span(std::ptrdiff_t num_elements): make_span{}, destroy_span{}, my_span{make_span(static_cast<size_t>(num_elements))}{ }

	inline unique_span(const unique_span<T,span_Type,constructor,deleter> &) = delete;
	inline unique_span & operator=(const unique_span<T,span_Type,constructor,deleter> &) = delete;

	inline unique_span(unique_span<T,span_Type,constructor,deleter> && in) noexcept : unique_span() { *this = std::move(in); }
	inline unique_span & operator=(unique_span<T,span_Type,constructor,deleter> && in) noexcept{
		using std::swap;
		swap(*this, in);
		return *this;
	}

	inline T& operator[](std::ptrdiff_t index) { return my_span[index]; }
	inline const T& operator[](std::ptrdiff_t index) const { return my_span[index]; }

	inline auto size() const noexcept{ return my_span.size(); }

	inline T* data() noexcept{ return my_span.data(); }
	inline const T* data() const noexcept{ return my_span.data(); }

	inline span_Type<T> span() noexcept{ return my_span; }
	inline span_Type<const T> span() const noexcept{ return my_span; }

	inline span_Type<T> subspan(std::ptrdiff_t offset, std::ptrdiff_t count) noexcept { return my_span.subspan(offset,count); }
	inline span_Type<const T> subspan(std::ptrdiff_t offset, std::ptrdiff_t count) const noexcept{ return my_span.subspan(offset,count); }

	inline span_Type<T> release() noexcept{
		span_Type<T> temp = my_span;
		my_span = span_Type<T>();
		return temp;
	}

	inline void reset() noexcept{
		destroy_span(my_span.data());
		my_span = span_Type<T>();
	}

	inline void reset(size_t num_elements) noexcept{
		destroy_span(my_span.data());
		my_span = make_span(num_elements);
	}

	inline ~unique_span() noexcept{ reset(); }

	template<typename T1, template<typename> class span_Type1, template<typename> class constructor1, template<typename> class deleter1>
	friend void swap(unique_span<T1,span_Type1,constructor1,deleter1> & lhs, unique_span<T1,span_Type1,constructor1,deleter1> & rhs) noexcept;

private:
	constructor<T> make_span;
	deleter<T> destroy_span;
	span_Type<T> my_span;
};

template<typename T, template<typename> class span_Type, template<typename> class constructor, template<typename> class deleter>
inline void swap(unique_span<T,span_Type,constructor,deleter> & lhs, unique_span<T,span_Type,constructor,deleter> & rhs) noexcept{
	using std::swap;
	swap(lhs.my_span,rhs.my_span);
}

template<typename T>
using dynamic_std_span = std::span<T>;

template<typename T>
using make_host_std_span = make_host_span<T,dynamic_std_span>;

template<typename T>
using unique_host_span = unique_span<T,dynamic_std_span,make_host_std_span,delete_host_span>;
/* ----- end unique_span wrapper ----- */

//ensures that std::copy knows at compile time that the pointers don't overlap <- might pull this out into a general utility function
//https://stackoverflow.com/questions/4707012/is-it-better-to-use-stdmemcpy-or-stdcopy-in-terms-to-performance
template <typename T1, typename T2>
inline void pcopy(T1 * __restrict out, const T2 * __restrict in, size_t n){ std::copy(in, in+n, out); }

} /* ----- end namespace PPP ----- */

#endif /* PPP_TYPES_HPP_ */
