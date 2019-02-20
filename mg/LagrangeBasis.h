#ifndef LAGRANGEBASIS_H
#define LAGRANGEBASIS_H

#include <LobattoQuadrature.h>

template <std::size_t n,bool projected=false>
class LagrangeBasis
{
public:

  constexpr LagrangeBasis(const double h_) :
    h(h_),
    paw(lobattoQuadrature<n>(h)),
    coeffs(std::array<double,n>())
  {}

  constexpr LagrangeBasis(const LagrangeBasis& lag) :
    h(lag.h),
    paw(lag.paw),
    coeffs(lag.coeffs)
  {}
  
  constexpr LagrangeBasis(const double h_,const std::array<double,n> & coeffs_) :
    h(h_),
    paw(lobattoQuadrature<n>(h)),
    coeffs(coeffs_)
  {}

  const double h ;
  const std::tuple<std::array<double,n>,
		   std::array<double,n> > paw ;
  const std::array<double,n> coeffs ;

  constexpr double operator () (int i, double x) const
  {
    const auto pts = std::get<0>(paw);
    double result = 1.;
    for (auto j=0u;j<n;++j)
      if (i != j) result *= ( x - pts[j]) / (pts[i] - pts[j]) ;
    return result ;
  }

  constexpr double operator () (double x) const
  {
    static_assert(projected,"The Lagrange basis has been instantiated without a coefficient vector, therefore, access is only granted to basis functions 'constexpr double LagrangeBasis::operator () (int i, double x) const'");
    double result = 0.;
    for (auto i=0u;i<n;++i)
      result += coeffs[i]*(*this)(i,x);
    return result;
  }

  constexpr double diff(int j, double x) const
  {
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
  constexpr double integrate(const Integrand & f) const
  {
    auto result = 0.;
    const auto pts = std::get<0>(paw);
    const auto wts = std::get<1>(paw);
    for (auto i = 0;i<pts.size();++i)
      result += f(pts[i])*wts[i] ;
    return result ;
  }

}; 

template <std::size_t p>
constexpr auto make_lagrange_basis(const double h)
{
  return LagrangeBasis<p+1,false>(h);
}

template <std::size_t p>
constexpr auto make_lagrange_proj(const double h,const auto & coeffs)
{
  return LagrangeBasis<p+1,true>(h,coeffs);
}

#endif /* LAGRANGEBASIS_H */
