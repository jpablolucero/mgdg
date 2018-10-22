#include <tuple>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
#include <blaze/Math.h>

template <std::size_t n,typename number=double>
constexpr auto assemble(const number d,const number L=1.)
{
  blaze::StaticMatrix<number,2*n,2*n> A{};
  const auto h = L/static_cast<number>(n);
  for (auto i = 0u;i<2*n;++i)
    {
      A(i,i) = d/h;
    }
  bool sw=true;
  for (auto i = 0u;i<2*n-2;++i)
    {
      A(i,i+2) = -1./(2.*h);
      if (sw) A(i+1,i+2) = -d/h+1./h;
      A(i+2,i) = -1./(2.*h);
      if (sw) A(i+2,i+1) = -d/h+1./h;
      if (sw) sw=false; else sw=true;
    }
  return A;
}

template <std::size_t n,typename number=double>
constexpr auto smoother_precond(const number d,const number L=1.)
{
  blaze::StaticMatrix<number,2*n,2*n> A{};
  const auto h = L/static_cast<number>(n);
  const double det = d/h*d/h - (-d/h+1./h)*(-d/h+1./h);
  A(0,0) = 1./(d/h);
  A(2*n-1,2*n-1) = 1./(d/h);
  for (auto i = 1u;i<2*n-1;++i)
    {
      A(i,i) = d/h / det ;
    }
  bool sw=true;
  for (auto i = 0u;i<2*n-2;++i)
    {
      if (sw) A(i+1,i+2) = (d/h-1./h)/det;
      if (sw) A(i+2,i+1) = (d/h-1./h)/det;
      if (sw) sw=false; else sw=true;
    }
  return A;
}

template <std::size_t n,typename number=double>
constexpr auto assemble_rhs()
{
  blaze::StaticVector<number,2*n> rhs{} ;
  for (auto & el : rhs) el = 1./static_cast<number>(n)/2.;
  return rhs;
}

constexpr std::size_t logg2(const std::size_t N)
{
  std::size_t res = 0;
  auto n = N;
  while (n != 0) {
    n /= 2;
    ++res;
  }
  return res;
}

template <std::size_t n,typename number=double>
constexpr blaze::StaticMatrix<number,n,2*n> restrict_matrix()
{
  blaze::StaticMatrix<number,n,2*n> R{};
  R(0,0) = 1.;
  R(0,1) = 0.5;
  R(1,1) = 0.5;
  R(0,2) = 0.5;
  R(1,2) = 0.5;

  auto i = 1u;
  for (auto j=3u;j<2*n-3;j+=4)
    {
      R(i,j) = 1.;
      R(i+1,j+1) = 1.;
      if (j<2*n-3)
	{
	  R(i+1,j+2) = 0.5;
	  R(i+2,j+2) = 0.5;
	  R(i+1,j+3) = 0.5;
	  R(i+2,j+3) = 0.5;
	}
      i+=2;
    }
  R(n-1,2*n-1) = 1.;

  return R;
}

template <std::size_t N, std::size_t ...Is>
constexpr auto make_restriction_impl(const std::index_sequence<Is...>)
{
  return std::make_tuple(restrict_matrix<(N >> Is)>()...);
}

template <std::size_t N,typename number=double>
constexpr auto make_restriction(blaze::StaticMatrix<number,N,N>)
{
  return make_restriction_impl<N/2>(std::make_index_sequence<logg2(N/2) - 1>());
}

template <std::size_t N,typename Restriction, typename number=double>
class Multigrid
{
public:
  
  constexpr Multigrid(const blaze::StaticMatrix<number,N,N> && A_,
		      const Restriction && rest_):
    A(std::forward<const blaze::StaticMatrix<number,N,N> & >(A_)),
    rest(std::forward<const Restriction>(rest_))
  {}

private:
  
  const blaze::StaticMatrix<number,N,N> & A ;
  const Restriction rest ;

public:
  
  template <std::size_t n0=2, std::size_t s=1, std::size_t m=1,std::size_t nr, std::size_t nc>
  constexpr blaze::StaticVector<number,nr>
  vcycle0(const blaze::StaticMatrix<number,nc,nr> && A,const blaze::StaticVector<number,nc> && res,
	  const number rlx=1.) const
  {
    if constexpr (nr == n0)
      {
	blaze::StaticVector<number,nr> q{},resq{};
	number norm = 1. ;
	while (true)
	  {
	    resq = res - A*q;
	    norm = blaze::norm(resq) ;
	    if (norm < 1.E-12) break ;      
	    q += (1./A(0,0)) * resq ;
	  }
	return q;
      }
    else
      {
	blaze::StaticVector<number,nr> x{};

	for (auto sit = 0;sit<s;++sit)
	  x += (rlx/A(0,0)) * (res - A*x);
      
	const auto & R = std::get<logg2(N/nr)-1>(rest);
	x += blaze::trans(R)*
	  vcycle0
	  <n0,m*s,m>(std::forward
		     <const blaze::StaticMatrix<number,nr/2,nr/2> >(R*A*blaze::trans(R)),
		     std::forward
		     <const blaze::StaticVector<number,nr/2> >(R * (res - A*x)),
		     rlx);
	                                 // q_1 = q_0 + RT A_0^{-1} (R (g - A x_1) - A_0 R q_0)
                                         //                    y_1 = x_1 + q_1

	for (auto sit = 0;sit<s;++sit)
	  x += (rlx/A(0,0)) * (res - A*x);

	return x ;
      }
  }
  
  template <std::size_t n0=2, std::size_t s=1, std::size_t m=1, bool prt=true,
	    std::size_t nr, std::size_t nc>
  constexpr blaze::StaticVector<number,nr>
  vcycle(const blaze::StaticMatrix<number,nc,nr> && A,
	 const blaze::StaticVector<number,nc> && res,
	 const number rlx=1.) const
  {
    if constexpr (nr == n0)
      {
	if constexpr (prt) std::cout << nr/2 << std::flush;
	blaze::StaticVector<number,nr> q{},resq{};
	number norm = 1. ;
	while (true)
	  {
	    resq = res - A*q;
	    norm = blaze::norm(resq) ;
	    if (norm < 1.E-12) break ;
	    q += vcycle0<2,1,2>
	      (std::forward<typename std::remove_reference<decltype(A)>::type> (A),
	       std::forward<typename std::remove_reference<decltype(resq)>::type>(resq));
	  }
	return q;
      }
    else
      {
	blaze::StaticVector<number,nr> x{};
	
	if constexpr (prt) std::cout << nr/2 << std::flush;
	if constexpr (prt and (s>1)) std::cout << "(" << s << ")" << std::flush ;

	for (auto sit = 0;sit<s;++sit)
	  x += (rlx/A(0,0))*(res - A*x);

	if constexpr (prt) std::cout << " \u2198 " << std::flush;
      
	const auto & R = std::get<logg2(N/nr)-1>(rest);
	x += blaze::trans(R)*
	  vcycle
	  <n0,m*s,m,prt>(std::forward
			 <const blaze::StaticMatrix<number,nr/2,nr/2> > (R*A*blaze::trans(R)),
			 std::forward
			 <const blaze::StaticVector<number,nr/2> >(R * (res - A*x)),
			 rlx);
                                         // q_1 = q_0 + RT A_0^{-1} (R (g - A x_1) - A_0 R q_0)
                                         //              y_1 = x_1 + q_1
      
	if constexpr (prt) std::cout << " \u2197 " <<  std::flush;

	for (auto sit = 0;sit<s;++sit)
	  x += (rlx/A(0,0))*(res - A*x);

	if constexpr (prt) std::cout << nr/2 << std::flush;
	if constexpr (prt and (s>1)) std::cout << "(" << s << ")" << std::flush ;

	return x ;
      }
  }

  template <bool prt=true>
  constexpr auto solve(const blaze::StaticVector<number,N> && res, double rlx = 1.) const
  {
    return vcycle<2,1,2,prt>
      (std::forward<typename std::remove_reference<decltype(A)>::type>(A),
       std::forward<typename std::remove_reference<decltype(res)>::type>(res),
       rlx);
  }

};

template <std::size_t nr, std::size_t nc, bool rnd = true, bool prt=true, bool rndtest=false,
	  typename number=double>
constexpr auto richardson(const blaze::StaticMatrix<number,nc,nr> & A,
			  const blaze::StaticVector<number,nr> & rhs,
			  const number accuracy)
{
  auto itm = 0. ;
  auto itt = 0u ;
  while(true)
    {
      blaze::StaticVector<number,nr> x{},res{};

      if constexpr (rnd)
	{
	  std::uniform_real_distribution<double> unif(0.,1.);
	  std::random_device re;
	  for (auto & el : x) el = unif(re);
	}
      
      res = rhs - A*x;                                    //                f - A x
      number initial_norm = blaze::norm(res) ;
      number norm = 1. ;
      auto it = 0u;
      const Multigrid M(std::forward<typename std::remove_reference<decltype(A)>::type>(A),
			make_restriction(A));
      while (true)
	{
	  res = rhs - A*x;                                //                f - A x
	  norm = blaze::norm(res) ;                       //             \| f - A x \|
	  if constexpr (prt) std::cout.precision(3);
	  if constexpr (prt) std::cout << "\rN:" << nr/2 << "\tIteration: "<< it << "\t"
	  			       << initial_norm << " \u2192 " << norm << " "
	   			       << std::scientific << std::flush;
	  if (norm/initial_norm < accuracy) break;        // break
	  ++it ;
	  x += M.template
	    solve(std::forward<typename std::remove_reference<decltype(res)>::type>(res));
	                                                  // x_1 = x_0 + B (f - A x_0)
	}
      
      if constexpr (!rndtest)
      		     {
      		       if constexpr (prt) std::cout << std::endl ;
      		       return x;
      		     }
      
      ++itt ;
      itm = itm * static_cast<double>(itt-1) / static_cast<double>(itt) +
  	static_cast<double>(it) / static_cast<double>(itt) ;
      if constexpr (prt) std::cout.precision(3);
      if constexpr (prt) std::cout << "\rIterations: "<< itm << std::flush ;
    }
  if constexpr (rndtest) return 0;
}


template <std::size_t nel,typename number=double>
constexpr auto mymain()
{
  return richardson<2*nel,2*nel>(assemble<nel>(1.2,1.),assemble_rhs<nel>(),1.E-8) ;
}

template <std::size_t N, std::size_t ...Is>
constexpr auto make_main_impl(const std::index_sequence<Is...>)
{
  return std::make_tuple(mymain<(N >> Is)>()...);
}

template <std::size_t N,typename number=double>
constexpr auto make_main()
{
  return make_main_impl<N>(std::make_index_sequence<logg2(N) - 1>());
}

int main(int argc, char *argv[])
{
  const volatile auto x = make_main<256>();
  return 0;
}
