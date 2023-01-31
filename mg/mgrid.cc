#include <Laplace.h>
#include <MFOperator.h>
#include <Solvers.h>

template <std::size_t n,std::size_t p=1,typename number=double>
constexpr auto assemble_rhs(const auto & g,const number L=1.)
{
  std::array<number,(p + 1)*n> rhs{} ;
  const auto h = L/static_cast<number>(n);
  const auto f = make_lagrange_basis<p>(h);
  const auto basis = make_lagrange_basis<p+3>(h);
  for (auto b = 0u ; b < n ; ++b)
    for (auto i = 0u ; i < p + 1 ; ++i)
      rhs[(p + 1)*b + i] =
	basis.integrate([&f,&g,i,b,h,L](const number x){return f(i,x)*g(x+b*h);});
	  
  return rhs;
}

template <std::size_t n,std::size_t p=1,typename number=double>
constexpr auto l2error(const auto & solution,const auto & g,const number L=1.)
{
  const auto h = L/static_cast<number>(n);
  const auto basis = make_lagrange_basis<p+3>(h);
  number result = 0. ;
  for (auto b = 0u ; b < n ; ++b)
    {
      std::array<number,(p + 1)> local_solution{} ;
      for (auto i = 0u ; i < p + 1 ; ++i)
	local_solution[i] = solution[(p + 1)*b + i] ;
      const auto f = make_lagrange_proj<p>(h,local_solution);
      result += basis.integrate
	([&f,&g,b,h,L](const number x){return (f(x)-g(x+b*h))*(f(x)-g(x+b*h));});
    }
  return std::sqrt(result);
}

template <std::size_t nel,std::size_t p=1,bool mf=false,typename number=double>
constexpr auto mymain()
{
  constexpr auto A =
    [&](){
      if constexpr
	(mf)
	  return MFOperator
	  (std::forward<LaplaceOperator<nel,p>>
	   (LaplaceOperator<nel,p>(static_cast<number>(p*(p+1)),1.)));
      else return MFOperator
	     (std::forward<std::array<std::array<number,(p+1)*nel>,(p+1)*nel>>
	      (laplaceMatrix<nel,p>(static_cast<number>(p*(p+1)),1.)));
    };
  constexpr auto rhs = assemble_rhs<nel,p>([](double x){return 1.;},1.);
  return richardson<p,false,mf,true>(A(),rhs,1.E-8);
}

template <std::size_t N,std::size_t p,std::size_t ...Is>
constexpr auto make_main_impl(const std::index_sequence<Is...>)
{
  return std::make_tuple(mymain<(N >> Is),p,false>()...);
}

template <std::size_t N,std::size_t p=1,typename number=double>
constexpr auto make_main()
{
  return make_main_impl<N,p>(std::make_index_sequence<logg2(N) - 1>());
}

int main(int argc,char *argv[])
{
  const volatile auto x1  = make_main<128,1>();
  // const volatile auto x1  = mymain<4,1>();
  return 0;
}
