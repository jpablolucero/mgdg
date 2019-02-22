#ifndef MULTIGRID_H
#define MULTIGRID_H

#include <EigenBindings.h>
#include <ArrayTools.h>
#include <Projections.h>

constexpr auto logg2(const auto N)
{
  auto res = 0u;
  auto n = N;
  while (n != 0)
    {
      n /= 2;
      ++res;
    }
  return res;
}
  
template <std::size_t... Is>
constexpr auto restrict(const auto & Rs, std::index_sequence<Is...>)
{
  return (std::get<sizeof...(Is) - Is - 1>(Rs) * ...);
}

template <std::size_t... Is>
constexpr auto prolongate(const auto & RTs, std::index_sequence<Is...>)
{
  return (std::get<Is>(RTs) * ...);
}

template <std::size_t... Is>
constexpr auto make_coarse_operators_impl(const auto & A, const auto & Rs, const auto & RTs,
					  std::index_sequence<Is...> is)
{
  return restrict(Rs, is) * A * prolongate(RTs, is);
}

template <std::size_t Index>
constexpr auto make_coarse_operators(const auto & A, const auto & Rs, const auto & RTs)
{
  return make_coarse_operators_impl(A, Rs, RTs, std::make_index_sequence<Index + 1>());
}

template <std::size_t N,std::size_t p,bool prt,
	  typename Operator,typename Smoother,typename Restriction,typename Prolongation,
	  typename number=double>
class Multigrid
{
public:
  
  constexpr Multigrid(const Operator && As_,
		      const Smoother && Ss_,
		      const Restriction && rest_,
		      const Prolongation && prol_):
    As(std::forward<const Operator>(As_)),
    Ss(std::forward<const Smoother>(Ss_)),
    rest(std::forward<const Restriction>(rest_)),
    prol(std::forward<const Prolongation>(prol_))
  {}

private:

  const Operator     As ;
  const Smoother     Ss ;
  const Restriction  rest ;
  const Prolongation prol ;

public:
  
  template <std::size_t n0=p+1,std::size_t s=1,std::size_t m=1>
  constexpr auto vcycle0(const auto && res,const number rlx=1.) const
  {
    const auto nr = std::size(res) ;
    const auto nc = std::size(res) ;
    const auto & A = std::get<logg2(N/nr)-1>(As);
    if constexpr (nr == n0)
      {
	typename std::remove_const<typename std::remove_reference<decltype(res)>::type>::type
	  q{},resq{};
	number norm = 1. ;
	while (true)
	  {
	    resq = res - A*q;                                                       
	    norm = std::sqrt(resq*resq) ;
	    if (norm < 1.E-12) break ;
	    q = q + invert<nr>(A)*resq ;
	  }
	return q;
      }
    else
      {
	typename std::remove_const<typename std::remove_reference<decltype(res)>::type>::type x{};
	const auto & B = std::get<logg2(N/nr)-1>(Ss);

	for (auto sit = 0;sit<s;++sit)
	  x = x + rlx * B * (res - A*x);           //    x_1 = x_0 + B (g - A x_0)

	const auto & R = std::get<logg2(N/nr)-1>(rest);
	const auto & RT = std::get<logg2(N/nr)-1>(prol);

	x = x + RT*vcycle0<n0,m*s,m>(std::forward<const decltype(R*(res-A*x))>(R*(res-A*x)),rlx);
	                                 // q_1 = q_0 + RT A_0^{-1} (R (g - A x_1) - A_0 R q_0)
                                         //                    y_1 = x_1 + q_1

	for (auto sit = 0;sit<s;++sit)
	  x = x + rlx * B * (res - A*x);           //    x_1 = x_0 + B (g - A x_0)

	return x ;
      }
  }
  
  template <std::size_t n0=p+1,std::size_t s=1,std::size_t m=1>
  constexpr auto vcycle(const auto && res,const number rlx=1.) const
  {
    const auto nr = std::size(res) ;
    const auto nc = std::size(res) ;
    const auto & A = std::get<logg2(N/nr)-1>(As);
    if constexpr (nr == n0)
      {
	if constexpr (prt) std::cout << nr/(p + 1) << std::flush;
	typename std::remove_const<typename std::remove_reference<decltype(res)>::type>::type
	  q{},resq{};
	number norm = 1. ;
	while (true)
	  {
	    resq = res - A*q;                                                       
	    norm = std::sqrt(resq*resq) ;
	    if (norm < 1.E-10) break ;
	    q = q + vcycle0<p+1,1,2>(std::forward<decltype(resq)>(resq),2./3.);
	  }
	return q;
      }
    else
      {
	typename std::remove_const<typename std::remove_reference<decltype(res)>::type>::type x{};
	const auto & B = std::get<logg2(N/nr)-1>(Ss);

	if constexpr (prt) std::cout << nr/(p + 1) << std::flush;
	if constexpr (prt and (s>1)) std::cout << "(" << s << ")" << std::flush ;

	for (auto sit = 0;sit<s;++sit)
	  x = x + rlx * B * (res - A*x);           //    x_1 = x_0 + B (g - A x_0)

	if constexpr (prt) std::cout << " \u2198 " << std::flush;
      
	const auto & R = std::get<logg2(N/nr)-1>(rest);
	const auto & RT = std::get<logg2(N/nr)-1>(prol);

	x = x + RT * vcycle<n0,m*s,m>(std::forward<const decltype(R*(res-A*x))>(R*(res-A*x)),rlx) ;
                                         // q_1 = q_0 + RT A_0^{-1} (R (g - A x_1) - A_0 R q_0)
                                         //              y_1 = x_1 + q_1
      
	if constexpr (prt) std::cout << " \u2197 " <<  std::flush;

	for (auto sit = 0;sit<s;++sit)
	  x = x + rlx * B * (res - A*x);           //    x_1 = x_0 + B (g - A x_0)

	if constexpr (prt) std::cout << nr/(p + 1) << std::flush;
	if constexpr (prt and (s>1)) std::cout << "(" << s << ")" << std::flush ;

	return x ;
      }
  }

  constexpr auto solve(const Eigen::Matrix<number,N,1> && res, number rlx=1.) const
  {
    std::array<double,N> rres{} ;
    for (auto i = 0u;i<N;++i)
      rres[i] = res(i) ;
    const auto result = vcycle<p+1,1,2>
      (std::forward<typename std::remove_reference<decltype(rres)>::type>(rres),rlx);
    return Eigenize(result);
  }

  constexpr auto solve(const auto && res, number rlx=1.) const
  {
    return vcycle<N/2,1,2>
      (std::forward<typename std::remove_reference<decltype(res)>::type>(res),rlx);
  }
};

template <std::size_t N,std::size_t p,std::size_t ...Is,typename number=double>
constexpr auto make_galerkin_impl(const std::index_sequence<Is...>)
{
  return std::make_tuple(laplaceMatrix<(N >> Is),p>(static_cast<number>(p*(p+1)),1.)...);
}

template <std::size_t N,std::size_t p=1,typename number=double>
constexpr auto make_galerkin()
{
  return make_galerkin_impl<N,p>(std::make_index_sequence<logg2(2*N)>());
}

template <std::size_t N,std::size_t p=1,std::size_t t=0,bool mf=false,bool gal=false,bool prt=true,
	  std::size_t... Is,typename number=double>
constexpr auto make_multigrid_impl(const auto & A, std::index_sequence<Is...>)
{
  const auto Rs = [&]() constexpr
    {
      if constexpr (mf) return std::make_tuple(MFOperator(Restrict<(N >> Is),p>())...);
      else return std::make_tuple(MFOperator(restrict_matrix<(N >> Is),p>())...);
    }();
  const auto RTs = [&]() constexpr
    {
      if constexpr (mf)
      {
	if constexpr (gal) return std::make_tuple(MFOperator(Prolongate<(N >> Is),p>(0.5))...);
	else return std::make_tuple(MFOperator(Prolongate<(N >> Is),p>())...);
      }
      else
	{
	  if constexpr(gal) return std::make_tuple(MFOperator(prolongate_matrix<(N >> Is),p>(0.5))...);
	  else return std::make_tuple(MFOperator(prolongate_matrix<(N >> Is),p>())...);
	}
    }();
  const auto As = [&]() constexpr
    {
      if constexpr (gal) return make_galerkin<N/(p+1),p>();
      else return std::make_tuple(A(), make_coarse_operators<Is>(A(), Rs, RTs)...);
    }();

  const auto Ss = std::make_tuple(block_jacobi<p+1,t,(N >> Is)>(std::get<Is>(As))...);
  return Multigrid<N,p,prt,
  		   typename std::remove_reference<decltype(As)>::type,
  		   typename std::remove_reference<decltype(Ss)>::type,
  		   typename std::remove_reference<decltype(Rs)>::type,
  		   typename std::remove_reference<decltype(RTs)>::type>
    (std::forward<typename std::remove_reference<decltype(As)>::type>(As),
     std::forward<typename std::remove_reference<decltype(Ss)>::type>(Ss),
     std::forward<typename std::remove_reference<decltype(Rs)>::type>(Rs),
     std::forward<typename std::remove_reference<decltype(RTs)>::type>(RTs));
}

template <std::size_t N,std::size_t p=1,std::size_t t=0,bool mf=false,bool gal=false,bool prt=true,
	  typename number=double>
constexpr auto make_multigrid(const auto & A)
{
  return make_multigrid_impl<N,p,t,mf,gal,prt>(A,std::make_index_sequence<logg2(N/(p+1))-1>());
}


#endif /* MULTIGRID_H */
