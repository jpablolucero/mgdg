#ifndef LAGRANGE_H
#define LAGRANGE_H

#include <lobatto.h>

template <std::size_t n>
class LagrangeBasis
{
public:

  constexpr LagrangeBasis(const double h_) :
    h(h_),
    paw(lobatto_compute<n>(h))
  {}

  const double h ;
  const std::tuple<std::array<double,n>,
		   std::array<double,n> > paw ;

  constexpr double operator () (int i, double x) const {
    const auto pts = std::get<0>(paw);
    double result = 1.;
    for (auto j=0u;j<n;++j)
      if (i != j) result *= ( x - pts[j]) / (pts[i] - pts[j]) ;
    return result ;
  }

  constexpr double diff(int j, double x) const {
    const auto pts = std::get<0>(paw);
    double sum = 0.;
    for (auto i = 0u ; i < n ; ++i)
      {
	if (i != j)
	  {
	    double mult = 1. / (pts[j] - pts[i]);
	    for (auto m=0u;m<n;++m)
	      if ((m != i)and(m != j)) mult *= ( x - pts[m]) / (pts[j] - pts[m]) ;
	    sum += mult ;
	  }
      }
    return sum ;
  }
  
  template <typename Integrand>
  constexpr double integrate(const Integrand & f) const {
    auto result = 0.;
    const auto pts = std::get<0>(paw);
    const auto wts = std::get<1>(paw);
    for (auto i = 0;i<pts.size();++i)
      result += f(pts[i])*wts[i] ;
    return result ;
  }

}; 

template <std::size_t p>
constexpr auto make_lagrange(const double h)
{
  return LagrangeBasis<p+1>(h);
}

#endif /* LAGRANGE_H */
