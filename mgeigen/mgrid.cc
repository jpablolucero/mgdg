#include <tuple>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>

#include <Eigen/Dense>
#include <unsupported/Eigen/IterativeSolvers>

#include <lagrange.h>
#include <arraytools.h>

template <int n,int p=1,typename number=double>
constexpr auto assemble(const number d,const number L=1.)
{
  Eigen::Matrix<number,p+1,p+1> CM{},CM0{},CM1{},FM{};
  CM.setZero() ; CM0.setZero() ; CM1.setZero() ; FM.setZero() ;
  const auto h = L/static_cast<double>(n);
  const auto f = make_lagrange<p>(h);
  const auto basis = make_lagrange<p+1>(h);
  for (auto i = 0u; i < p + 1; ++i)
    for (auto j = 0u; j < p + 1; ++j)
      {
	CM(i,j) = basis.integrate([&f,i,j](const double x){return f.diff(i,x) * f.diff(j,x);})
	  - (0 - f(i,0.)) * (0 + f.diff(j,0.)) / 2.
	  - (0 + f.diff(i,0.)) / 2. * (0 - f(j,0.))
	  - (f(i,h) - 0) * (f.diff(j,h) + 0) / 2
	  - (f.diff(i,h) + 0) / 2. * (f(j,h) - 0)
	  + d / h * (0 - f(i,0.)) * (0 - f(j,0.))
	  + d / h * (f(i,h) - 0) * (f(j,h) - 0);
	CM0(i,j) = CM(i,j) ;
	CM1(i,j) = CM(i,j) ;
	if ((i==0)or(j==0))
	  CM0(i,j) += - (0 - f(i,0.)) * (0 + f.diff(j,0.)) / 2
	    - (0 + f.diff(i,0.)) / 2 * (0 - f(j,0.))
	    + d / h * (0 - f(i,0.)) * (0 - f(j,0.)) ;
	if ((i==p)or(j==p))
	  CM1(i,j) += - (f(i,h) - 0) * (f.diff(j,h) + 0 ) / 2
	    - (f.diff(i,h) + 0) / 2 * (f(j,h) - 0)
	    + d / h * (f(i,h) - 0) * (f(j,h) - 0);

	FM(i,j) = - (f(i,h) - 0) * (0 + f.diff(j,0.)) / 2
	  - (f.diff(i,h) + 0) / 2 * (0 - f(j,0.))
	  + d / h * (f(i,h) - 0) * (0 - f(j,0.));
      }

  Eigen::Matrix<number,(p+1)*n,(p+1)*n> A{};
  A.setZero() ;
  for (auto b = 0u ; b < n ; ++b)
    for (auto i = 0u ; i < p + 1 ; ++i)
      for (auto j = 0u ; j < p + 1 ; ++j)
	{
	  if (b==0) A((p + 1)*b+i,(p + 1)*b+j) = CM0(i,j) ;
	  else if (b==n-1) A((p + 1)*b+i,(p + 1)*b+j) = CM1(i,j) ;
	  else A((p + 1)*b+i,(p + 1)*b+j) = CM(i,j) ;
	  if (b != n - 1) A((p + 1)*b+i,(p + 1)*b+j+(p + 1)) = FM(i,j) ;
	  if (b != 0) A((p + 1)*b+i,(p + 1)*b+j-(p + 1)) = FM(j,i) ;
	}

  return A;
}

template <int n,int p=1,typename number=double>
constexpr auto assemble_rhs(const number L=1.)
{
  Eigen::Matrix<number,(p + 1)*n,1> rhs{} ;
  rhs.setZero() ;
  const auto h = L/static_cast<double>(n);
  const auto f = make_lagrange<p>(h);
  const auto basis = make_lagrange<p+1>(h);
  for (auto b = 0u ; b < n ; ++b)
    for (auto i = 0u ; i < p + 1 ; ++i)
      rhs[(p + 1)*b + i] = basis.integrate([&f,i](const double x){return f(i,x);});
  return rhs;
}

constexpr int logg2(const int N)
{
  int res = 0;
  auto n = N;
  while (n != 0) {
    n /= 2;
    ++res;
  }
  return res;
}

template <int N,int p=1,typename number=double>
constexpr Eigen::Matrix<number,N,2*N> restrict_matrix()
{
  Eigen::Matrix<number,N,2*N> R{};
  R.setZero() ;
  if (p == 1)
    {
      R(0,0) = 1.;
      R(0,1) = 0.5;
      R(1,1) = 0.5;
      R(0,2) = 0.5;
      R(1,2) = 0.5;

      auto i = 1u;
      for (auto j=3u;j<2*N-3;j+=4)
	{
	  R(i,j) = 1.;
	  R(i+1,j+1) = 1.;
	  if (j<2*N-3)
	    {
	      R(i+1,j+2) = 0.5;
	      R(i+2,j+2) = 0.5;
	      R(i+1,j+3) = 0.5;
	      R(i+2,j+3) = 0.5;
	    }
	  i+=2;
	}
      R(N-1,2*N-1) = 1.;
    }
  else
    {
      auto n = N/(p + 1) ;
      for (auto j = 0 ; j < n ; ++j)
	for (auto i = 0 ; i < p ; ++i)
	  {
	    R(0+i+j*(p+1),0+i*(p+1)+2*j*(p+1)) = 1.;
	    R(0+i+j*(p+1),1+i*(p+1)+2*j*(p+1)) = 0.5;
	    R(1+i+j*(p+1),1+i*(p+1)+2*j*(p+1)) = 0.5;
	    R(1+i+j*(p+1),2+i*(p+1)+2*j*(p+1)) = 1.;
	  }
    }

  return R;
}

template <int N,int p=1,int ...Is>
constexpr auto make_restriction_impl(const std::integer_sequence<int,Is...>)
{
  return std::make_tuple(restrict_matrix<(N >> Is),p>()...);
}

template <int N,int p=1,typename number=double>
constexpr auto make_restriction()
{
  return make_restriction_impl<N/2,p>(std::make_integer_sequence<int,logg2(N/2) - 1>());
}

template <int N, typename Restriction, typename number=double>
class Multigrid
{
public:
  
  constexpr Multigrid(const Eigen::Matrix<number,N,N> && A_,
		      const Restriction && rest_):
    A(std::forward<const Eigen::Matrix<number,N,N> & >(A_)),
    rest(std::forward<const Restriction>(rest_))
  {}

private:

  const Eigen::Matrix<number,N,N> & A ;
  const Restriction rest ;

public:
  
  template <int p=1, int n0=p+1, int s=1, int m=1,int t=0,
	    int nr, int nc>
  constexpr Eigen::Matrix<number,nr,1>
  vcycle0(const Eigen::Matrix<number,nc,nr> && A,const Eigen::Matrix<number,nc,1> && res,
	  const number rlx=1.) const
  {
    if constexpr (nr == n0)
      {
	Eigen::Matrix<number,nr,1> q{},resq{};
	q.setZero();
	resq.setZero();
	number norm = 1. ;
	while (true)
	  {
	    resq = res - A*q;
	    norm = resq.norm() ;
	    if (norm < 1.E-12) break ;      
	    q += A.inverse() * resq ;
	  }
	return q;
      }
    else
      {
	Eigen::Matrix<number,nr,1> x{};
	x.setZero();
	const auto B = block_jacobi<p+1,t>(A) ;

	for (auto sit = 0;sit<s;++sit)
	  x += rlx * B * (res - A*x);                       //    x_1 = x_0 + B (g - A x_0)
      
	const auto & R = std::get<logg2(N/nr)-1>(rest);
	x += R.transpose()*
	  vcycle0
	  <p,n0,m*s,m,t>(std::forward<const Eigen::Matrix<number,nr/2,nr/2> >(R*A*R.transpose()),
			 std::forward<const Eigen::Matrix<number,nr/2,1> >(R * (res - A*x)),
			 rlx);
	                                 // q_1 = q_0 + RT A_0^{-1} (R (g - A x_1) - A_0 R q_0)
                                         //                    y_1 = x_1 + q_1

	for (auto sit = 0;sit<s;++sit)
	  x += rlx * B * (res - A*x);                       //    y_2 = y_1 + B (g - A y_1)

	return x ;
      }
  }

  template <int p=1, int n0=p+1, int s=1, int m=1, int t=0,
	    bool prt=true, int nr, int nc>
  constexpr Eigen::Matrix<number,nr,1>
  vcycle(const Eigen::Matrix<number,nc,nr> && A,const Eigen::Matrix<number,nc,1> && res,
	 const number rlx=1.) const
  {
    if constexpr (nr == n0)
      {
	if constexpr (prt) std::cout << nr/(p + 1) << std::flush;
	Eigen::Matrix<number,nr,1> q{},resq{};
	q.setZero();
	resq.setZero();
	number norm = 1. ;
	while (true)
	  {
	    resq = res - A*q;
	    norm = resq.norm() ;
	    if (norm < 1.E-12) break ;
	    q += vcycle0<p,n0,1,2>
	      (std::forward<typename std::remove_reference<decltype(A)>::type> (A),
	       std::forward<typename std::remove_reference<decltype(resq)>::type>(resq));
	  }
	return q;
      }
    else
      {
	Eigen::Matrix<number,nr,1> x{};
	x.setZero();
	const auto B = block_jacobi<p+1,t>(A) ;

	if constexpr (prt) std::cout << nr/(p + 1) << std::flush;
	if constexpr (prt and (s>1)) std::cout << "(" << s << ")" << std::flush ;

	for (auto sit = 0;sit<s;++sit)
	  x += rlx * B * (res - A*x);                       //    x_1 = x_0 + B (g - A x_0)

	if constexpr (prt) std::cout << " \u2198 " << std::flush;
	
	const auto & R = std::get<logg2(N/nr)-1>(rest);

	x += R.transpose() *
	  vcycle
	  <p,n0,m*s,m,t,prt>(std::forward
			     <const Eigen::Matrix<number,nr/2,nr/2> >(R*A*R.transpose()),
			     std::forward
			     <const Eigen::Matrix<number,nr/2,1> >(R * (res - A*x)),
			     rlx);
	                                 // q_1 = q_0 + RT A_0^{-1} (R (g - A x_1) - A_0 R q_0)
                                         //                    y_1 = x_1 + q_1
      
	if constexpr (prt) std::cout << " \u2197 " <<  std::flush;

	for (auto sit = 0;sit<s;++sit)
	  x += rlx * B *(res - A*x);                       //    y_2 = y_1 + B (g - A y_1)

	if constexpr (prt) std::cout << nr/(p + 1) << std::flush;
	if constexpr (prt and (s>1)) std::cout << "(" << s << ")" << std::flush ;

	return x ;
      }
  }

  template <int p=1,bool prt=true>
  constexpr auto solve(const Eigen::Matrix<number,N,1> && res, double rlx = 1.) const
  {
    return vcycle<p,p+1,1,2,0,prt>
      (std::forward<typename std::remove_reference<decltype(A)>::type>(A),
       std::forward<typename std::remove_reference<decltype(res)>::type>(res),
       rlx);
  }
  
};

template <int nr, int nc, int p=1,
	  bool rnd = true, bool prt=true, bool rndtest=false,typename number=double>
constexpr auto richardson(const Eigen::Matrix<number,nc,nr> & A,
			  const Eigen::Matrix<number,nr,1> & rhs,
			  const number accuracy)
{
  auto itm = 0. ;
  auto itt = 0u ;
  while(true)
    {
      Eigen::Matrix<number,nr,1> x{},res{};
      x.setZero();
      res.setZero();

      if constexpr (rnd)
	{
	  std::uniform_real_distribution<double> unif(0.,1.);
	  std::random_device re;
	  for (auto & el : x) el = unif(re);
	}
      
      res = rhs - A*x;                                    //                f - A x
      number initial_norm = res.norm() ;
      number norm = 1. ;
      auto it = 0u;
      const Multigrid M(std::forward<typename std::remove_reference<decltype(A)>::type>(A),
			make_restriction<nc,p>());
      while (true)
	{
	  res = rhs - A*x;                                //                f - A x
	  norm = res.norm() ;                             //             \| f - A x \|
	  if constexpr (prt) std::cout.precision(3);
	  if constexpr (prt) std::cout << "\rN:" << std::left << std::setw(5) << nr/(p+1)
				       << "p:" << std::left << std::setw(3) << p
				       << std::flush ;
	  if constexpr (prt) std::cout << " Iteration: " << std::left << std::setw(7)
				       << it << std::flush;
	  if constexpr (prt) std::cout << initial_norm << " \u2192 " << norm << " "
	   			       << std::scientific << std::flush ;
	  if (norm/initial_norm < accuracy) break;        // break

	  ++it ;
	  x += M.template
	    solve<p,prt>(std::forward<typename std::remove_reference<decltype(res)>::type>(res));
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
      if constexpr (rndtest) std::cout.precision(3);
      if constexpr (rndtest) std::cout << "\rIterations: "<< itm << std::flush ;
    }
  if constexpr (rndtest) return 0;
}

template <int nr, int nc, int p=1,
	  bool rnd = true, bool prt=true, bool rndtest=false,typename number=double>
constexpr auto gmres(const Eigen::Matrix<number,nc,nr> & A,
		     const Eigen::Matrix<number,nr,1> & rhs,
		     const number accuracy)
{
  auto itm = 0. ;
  auto itt = 0u ;
  while(true)
    {
      Eigen::Matrix<number,nr,1> x{},res{};
      x.setZero();
      res.setZero();

      if constexpr (rnd)
	{
	  std::uniform_real_distribution<double> unif(0.,1.);
	  std::random_device re;
	  for (auto & el : x) el = unif(re);
	}
      
      res = rhs - A*x;                                    //                f - A x
      number initial_norm = res.norm() ;
      number norm = 1. ;
      auto it = 0u;
      const Multigrid M(std::forward<typename std::remove_reference<decltype(A)>::type>(A),
			make_restriction<nc,p>());
      if constexpr (prt) std::cout.precision(3);
      if constexpr (prt) std::cout << "\rN:" << std::left << std::setw(5) << nr/(p+1)
				   << "p:" << std::left << std::setw(3) << p
				   << std::flush ;
      if constexpr (prt) std::cout << " Iteration: " << std::left << std::setw(7)
				   << it << std::flush;
      if constexpr (prt) std::cout << initial_norm << " \u2192 " << norm << " "
				   << std::scientific << std::flush ;
      long int iterations = 30;
      long int & its = iterations;
      double tolerance = accuracy;
      double & tol = tolerance;
      const auto gmres_res = Eigen::internal::gmres(A,rhs,x,M,its,30,tol);
      it = its;
      
      if constexpr (!rndtest)
      		     {
      		       if constexpr (prt) std::cout << std::endl ;
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

template <int nel,int p=1,typename number=double>
constexpr auto mymain()
{
  return richardson<nel*(p+1),nel*(p+1),p,false>
    (assemble<nel,p>(static_cast<double>(p*(p+1)),1.),assemble_rhs<nel,p>(1.),1.E-8) ;
}

template <int N,int p=1,int ...Is>
constexpr auto make_main_impl(const std::integer_sequence<int,Is...>)
{
  return std::make_tuple(mymain<(N >> Is),p>()...);
}

template <int N,int p=1,typename number=double>
constexpr auto make_main()
{
  return make_main_impl<N,p>(std::make_integer_sequence<int,logg2(N) - 1>());
}

int main(int argc, char *argv[])
{
  const volatile auto x = make_main<128,2>();
  
  return 0;
}
