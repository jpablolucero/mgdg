#include <array>
#include <tuple>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <ios>
#include <random>
#include <thread>

#include <Eigen/Dense>
#include <unsupported/Eigen/IterativeSolvers>

#include <arraytools.h>
#include <lagrange.h>

template <std::size_t n,std::size_t p=1,typename number=double>
constexpr auto assemble(const number d=static_cast<number>(p*(p+1)),const number L=1.)
{
  std::array<std::array<number,p+1>,p+1> CM{},CM0{},CM1{},FM{};
  const auto h = L/static_cast<number>(n);
  const auto f = make_lagrange<p>(h);
  const auto basis = make_lagrange<p+1>(h);

  for (auto i = 0u; i < p + 1; ++i)
    for (auto j = 0u; j < p + 1; ++j)
      {
	CM[i][j] = basis.integrate([&f,i,j](const number x){return f.diff(i,x) * f.diff(j,x);})
	  - (0 - f(i,0.)) * (0 + f.diff(j,0.)) / 2. 
	  - (0 + f.diff(i,0.)) / 2. * (0 - f(j,0.))
	  - (f(i,h) - 0) * (f.diff(j,h) + 0) / 2 
	  - (f.diff(i,h) + 0) / 2. * (f(j,h) - 0)
	  + d / h * (0 - f(i,0.)) * (0 - f(j,0.))
	  + d / h * (f(i,h) - 0) * (f(j,h) - 0);
	CM0[i][j] = CM[i][j] ;
	CM1[i][j] = CM[i][j] ;
	if ((i==0)or(j==0))
	  CM0[i][j] += - (0 - f(i,0.)) * (0 + f.diff(j,0.)) / 2
	    - (0 + f.diff(i,0.)) / 2 * (0 - f(j,0.))
	    + d / h * (0 - f(i,0.)) * (0 - f(j,0.)) ;
	if ((i==p)or(j==p))
	  CM1[i][j] += - (f(i,h) - 0) * (f.diff(j,h) + 0 ) / 2 
	    - (f.diff(i,h) + 0) / 2 * (f(j,h) - 0)
	    + d / h * (f(i,h) - 0) * (f(j,h) - 0);
	    
	FM[i][j] = - (f(i,h) - 0) * (0 + f.diff(j,0.)) / 2
	  - (f.diff(i,h) + 0) / 2 * (0 - f(j,0.))
	  + d / h * (f(i,h) - 0) * (0 - f(j,0.));
      }

  std::array<std::array<number,(p+1)*n>,(p+1)*n> A{};
  if constexpr (n == 1)
		 for (auto i = 0u ; i < p + 1 ; ++i)
		   for (auto j = 0u ; j < p + 1 ; ++j)
		     {
		       if ((i==0)or(j==0)) A[i][j] = CM0[i][j] ;
		       else if ((i==p)or(j==p)) A[i][j] = CM1[i][j] ;
		       else A[i][j] = CM[i][j] ;
		     }
  else 
    for (auto b = 0u ; b < n ; ++b)
      for (auto i = 0u ; i < p + 1 ; ++i)
	for (auto j = 0u ; j < p + 1 ; ++j)
	  {
	    if (b==0) A[(p + 1)*b+i][(p + 1)*b+j] = CM0[i][j] ;
	    else if (b==n-1) A[(p + 1)*b+i][(p + 1)*b+j] = CM1[i][j] ;
	    else A[(p + 1)*b+i][(p + 1)*b+j] = CM[i][j] ;
	    if (b != n - 1) A[(p + 1)*b+i][(p + 1)*b+j+(p + 1)] = FM[i][j] ;
	    if (b != 0) A[(p + 1)*b+i][(p + 1)*b+j-(p + 1)] = FM[j][i] ;
	  }
  return A;
}

template <std::size_t n,std::size_t p=1,typename number=double>
constexpr auto assemble_rhs(const number L=1.)
{
  std::array<number,(p + 1)*n> rhs{} ;
  const auto h = L/static_cast<number>(n);
  const auto f = make_lagrange<p>(h);
  const auto basis = make_lagrange<p+1>(h);
  for (auto b = 0u ; b < n ; ++b)
    for (auto i = 0u ; i < p + 1 ; ++i)
      rhs[(p + 1)*b + i] = basis.integrate([&f,i](const number x){return f(i,x);});
	  
  return rhs;
}

constexpr auto logg2(const auto N)
{
  auto res = 0u;
  auto n = N;
  while (n != 0) {
    n /= 2;
    ++res;
  }
  return res;
}

template <std::size_t N,std::size_t p=1,typename number=double>
constexpr auto restrict_matrix(const double factor = 1.)
{
  std::array<std::array<number,N>,N/2> R{};
  for (auto k = 0u ; k < N/2 ; k += p + 1)
    {
      bool avg = false ;
      std::size_t j = k ;
      for (auto i = 2*k ; i < 2*k+p+1 ; ++i)
	{
	  if (avg)
	    {
	      R[j][i] = 0.5 * factor ;
	      R[j+1][i] = 0.5 * factor ;
	      j += 1 ;
	      avg = false ;
	    }
	  else
	    {
	      R[j][i] = 1. * factor ;
	      avg = true ;
	    }
	}
      if (avg) avg = false ; else { avg = true ; j -= 1 ; }
      for (auto i = 2*k+p+1 ; i < 2*k+2*(p+1) ; ++i)
	{
	  if (avg)
	    {
	      R[j][i] = 0.5 * factor ;
	      R[j+1][i] = 0.5 * factor ;
	      j += 1 ;
	      avg = false ;
	    }
	  else
	    {
	      R[j][i] = 1. * factor ;
	      avg = true ;
	    }
	}
    }
  return R;
}

template <std::size_t N,std::size_t p=1,typename number=double>
constexpr auto prolongate_matrix(const double factor = 1.)
{
  std::array<std::array<number,N/2>,N> RT{};
  for (auto k = 0u ; k < N/2 ; k += p + 1)
    {
      bool avg = false ;
      std::size_t j = k ;
      for (auto i = 2*k ; i < 2*k+p+1 ; ++i)
	{
	  if (avg)
	    {
	      RT[i][j] = 0.5 * factor ;
	      RT[i][j+1] = 0.5 * factor ;
	      j += 1 ;
	      avg = false ;
	    }
	  else
	    {
	      RT[i][j] = 1. * factor ;
	      avg = true ;
	    }
	}
      if (avg) avg = false ; else { avg = true ; j -= 1 ; }
      for (auto i = 2*k+p+1 ; i < 2*k+2*(p+1) ; ++i)
	{
	  if (avg)
	    {
	      RT[i][j] = 0.5 * factor ;
	      RT[i][j+1] = 0.5 * factor ;
	      j += 1 ;
	      avg = false ;
	    }
	  else
	    {
	      RT[i][j] = 1. * factor ;
	      avg = true ;
	    }
	}
    }
  return RT;
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

template <std::size_t N,std::size_t M,typename number=double>
Eigen::Matrix<number,N,M> Eigenize(std::array<std::array<number,M>,N> data)
{
  Eigen::Matrix<number,N,M> eMatrix;
  eMatrix.setZero();
  for (int i = 0; i < data.size(); ++i)
    eMatrix.row(i) = Eigen::VectorXd::Map(&data[i][0], data[0].size());
  return eMatrix;
}

template <std::size_t M,typename number=double>
Eigen::Matrix<number,M,1> Eigenize(std::array<number,M> data)
{
  Eigen::Matrix<number,M,1> eMatrix(data.size(), 1);
  eMatrix.setZero();
  eMatrix.col(0) = Eigen::VectorXd::Map(&data[0], data.size());
  return eMatrix;
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
	    q = q + invert(A)*resq ;
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
  return std::make_tuple(assemble<(N >> Is),p>(static_cast<number>(p*(p+1)),1.)...);
}

template <std::size_t N,std::size_t p=1,typename number=double>
constexpr auto make_galerkin()
{
  return make_galerkin_impl<N,p>(std::make_index_sequence<logg2(2*N)>());
}

template <std::size_t p=1,std::size_t t=0,bool gal=false,bool prt=true,std::size_t... Is,
	  typename number=double>
constexpr auto make_multigrid_impl(const auto & A, std::index_sequence<Is...>)
{
  const std::size_t N = std::size(A);
  const auto Rs = std::make_tuple(restrict_matrix<(N >> Is),p>()...);
  const auto RTs = [&]() constexpr
    {
      if constexpr (gal) return std::make_tuple(prolongate_matrix<(N >> Is),p>(0.5)...);
      else return std::make_tuple(prolongate_matrix<(N >> Is),p>()...);
    }();
  const auto As = [&]() constexpr
    {
      if constexpr (gal) return make_galerkin<N/(p+1),p>();
      else return std::make_tuple(A, make_coarse_operators<Is>(A, Rs, RTs)...);
    }();
  const auto Ss = std::make_tuple(block_jacobi<p+1,t>(std::get<Is>(As))...);
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

template <std::size_t p=1,std::size_t t=0,bool gal=false,bool prt=true,typename number=double>
constexpr auto make_multigrid(const auto & A)
{
  return make_multigrid_impl<p,t,gal,prt>(A,std::make_index_sequence<logg2(std::size(A)/(p+1))-1>());
}

template <std::size_t p=1,bool rnd=true,bool prt=true,bool rndtest=false,typename number=double>
constexpr auto richardson(const auto & A,const auto & rhs,const number accuracy)
{
  auto itm = 0. ;
  auto itt = 0u ;
  while(true)
    {
      typename std::remove_const<typename std::remove_reference<decltype(rhs)>::type>::type x{},res{};
      if constexpr (rnd)
  	{
  	  std::uniform_real_distribution<number> unif(0.,1.);
  	  std::random_device re;
  	  for (auto & el : x) el = unif(re);
  	}

      res = rhs - A*x ;                                   //                f - A x
      number initial_norm = std::sqrt(res*res) ;
      number norm = 1. ;
      auto it = 0u;
      const auto M = make_multigrid<p,0,true,prt>(A);
      const number rlx = 1. ;
      while (true)
      	{
      	  res = rhs - A*x;                                //                f - A x
      	  norm = std::sqrt(res*res) ;                     //             \| f - A x \|
      	  if constexpr (prt) std::cout.precision(3);
      	  if constexpr (prt) std::cout << "\rN:" << std::left << std::setw(5) << std::size(A)/(p+1)
      	  			       << "p:" << std::left << std::setw(3) << p
      	  			       << "rlx:" << std::left << std::setw(10) << rlx
      	  			       << std::flush ;
      	  if constexpr (prt) std::cout << " Iteration: " << std::left << std::setw(7)
      	  			       << it << std::flush;
      	  if constexpr (prt) std::cout << initial_norm << " \u2192 " << norm << " "
      	   			       << std::scientific << std::flush ;
      	  if (norm/initial_norm < accuracy) break;        // break
      	  ++it ;
	  x = x + M.solve(std::forward<typename std::remove_reference<decltype(res)>::type>(res),rlx);
      	                                                  // x_1 = x_0 + B (f - A x_0)
	  if (it == 101) break ;
      	}
      
      if constexpr (!rndtest)
      		     {
      		       if constexpr (prt) std::cout << std::endl ;
  		       return x;
  		     }
      ++itt ;
      itm = itm * static_cast<number>(itt-1) / static_cast<number>(itt) +
      	static_cast<number>(it) / static_cast<number>(itt) ;
      if constexpr (rndtest) std::cout.precision(3);
      if constexpr (rndtest) std::cout << "\rIterations: "<< itm << std::flush ;
    }
  if constexpr (rndtest) return 0;
}

template <int p=1,bool rnd=true,bool prt=false,bool rndtest=false,typename number=double>
constexpr auto gmres(const auto & AA,const auto & rrhs,const number accuracy)
{
  const auto A = Eigenize(AA);
  const auto rhs = Eigenize(rrhs);
  auto itm = 0. ;
  auto itt = 0u ;
  while(true)
    {
      Eigen::Matrix<number,std::size(rrhs),1> x{},res{};
      x.setZero();
      res.setZero();
      if constexpr (rnd)
	{
	  std::uniform_real_distribution<double> unif(0.,1.);
	  std::random_device re;
	  for (auto & el : x) el = unif(re);
	}

      auto it = 0u;

      const auto M = make_multigrid<p,(p+1)/2,false>(AA);
      long int iterations = 100;
      long int & its = iterations;
      double tolerance = accuracy;
      double & tol = tolerance;
      const number rlx = 1. ;
      const auto gmres_res = Eigen::internal::gmres(A,rhs,x,M,its,100,tol,p,rlx);

      it = its;
      
      if constexpr (!rndtest)
      		     {
		       std::cout << std::endl ;
      		       return x;
      		     }
      ++itt ;
      itm = itm * static_cast<double>(itt-1) / static_cast<double>(itt) +
  	static_cast<double>(it) / static_cast<double>(itt) ;
      if constexpr (rndtest) std::cout.precision(3);
      if constexpr (rndtest) std::cout << "\rIterations: "<< itm << std::flush ;
    }
  if constexpr (rndtest) return 0;
}

template <std::size_t nel,std::size_t p=1,typename number=double>
constexpr auto mymain()
{
  return richardson<p,false>
    (assemble<nel,p>(static_cast<number>(p*(p+1)),1.),assemble_rhs<nel,p>(1.),1.E-8) ;
}

template <std::size_t N,std::size_t p,std::size_t ...Is>
constexpr auto make_main_impl(const std::index_sequence<Is...>)
{
  return std::make_tuple(mymain<(N >> Is),p>()...);
}

template <std::size_t N,std::size_t p=1,typename number=double>
constexpr auto make_main()
{
  return make_main_impl<N,p>(std::make_index_sequence<logg2(N) - 1>());
}

int main(int argc,char *argv[])
{
  const volatile auto x1  = mymain<4,2>();
  return 0;
}
