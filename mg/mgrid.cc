#include <array>
#include <tuple>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <ios>
#include <random>

#include <arraytools.h>
#include <lagrange.h>

template <std::size_t n,std::size_t p=1,typename number=double>
constexpr auto assemble(const number d,const number L=1.)
{
  std::array<std::array<number,p+1>,p+1> CM{},CM0{},CM1{},FM{};
  const auto h = L/static_cast<double>(n);
  const auto f = make_lagrange<p>(h);
  const auto basis = make_lagrange<p+1>(h);
  for (auto i = 0u; i < p + 1; ++i)
    for (auto j = 0u; j < p + 1; ++j)
      {
	CM[i][j] = basis.integrate([&f,i,j](const double x){return f.diff(i,x) * f.diff(j,x);})
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

template <std::size_t n,typename number=double>
constexpr auto smoother_precond(const number d,const number L=1.)
{
  std::array<std::array<number,2*n>,2*n> A{};
  const auto h = L/static_cast<number>(n);
  const double det = d/h*d/h - (-d/h+1./h)*(-d/h+1./h);
  A[0][0] = 1./(d/h);
  A[2*n-1][2*n-1] = 1./(d/h);
  for (auto i = 1u;i<2*n-1;++i)
    {
      A[i][i] = d/h / det ;
    }
  bool sw=true;
  for (auto i = 0u;i<2*n-2;++i)
    {
      if (sw) A[i+1][i+2] = (d/h-1./h)/det;
      if (sw) A[i+2][i+1] = (d/h-1./h)/det;
      if (sw) sw=false; else sw=true;
    }
  return A;
}

template <std::size_t n,std::size_t p=1,typename number=double>
constexpr auto assemble_rhs(const number L=1.)
{
  std::array<number,(p + 1)*n> rhs{} ;
  const auto h = L/static_cast<double>(n);
  const auto f = make_lagrange<p>(h);
  const auto basis = make_lagrange<p+1>(h);
  for (auto b = 0u ; b < n ; ++b)
    for (auto i = 0u ; i < p + 1 ; ++i)
      rhs[(p + 1)*b + i] = basis.integrate([&f,i](const double x){return f(i,x);});
	  
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

template <std::size_t N,std::size_t p=1,typename number=double>
constexpr std::array<std::array<number,2*N>,N> restrict_matrix()
{
  std::array<std::array<number,2*N>,N> R{};
  for (auto k = 0u ; k < N ; k += p + 1)
    {
      bool avg = false ;
      std::size_t j = k ;
      for (auto i = 2*k ; i < 2*k+p+1 ; ++i)
	{
	  if (avg)
	    {
	      R[j][i] = 0.5 ;
	      R[j+1][i] = 0.5 ;
	      j += 1 ;
	      avg = false ;
	    }
	  else
	    {
	      R[j][i] = 1. ;
	      avg = true ;
	    }
	}
      if (avg) avg = false ; else { avg = true ; j -= 1 ; }
      for (auto i = 2*k+p+1 ; i < 2*k+2*(p+1) ; ++i)
	{
	  if (avg)
	    {
	      R[j][i] = 0.5 ;
	      R[j+1][i] = 0.5 ;
	      j += 1 ;
	      avg = false ;
	    }
	  else
	    {
	      R[j][i] = 1. ;
	      avg = true ;
	    }
	}
    }
  return R;
}

template <std::size_t N,std::size_t p=1,std::size_t ...Is>
constexpr auto make_restriction_impl(const std::index_sequence<Is...>)
{
  return std::make_tuple(restrict_matrix<(N >> Is) * (p + 1) / 2,p>()...);
}

template <std::size_t N,std::size_t p=1,typename number=double>
constexpr auto make_restriction()
{
  return make_restriction_impl<N/2,p>(std::make_index_sequence<logg2(N/2) - 1>());
}

template <std::size_t N,std::size_t p=1,typename number=double>
constexpr std::array<std::array<number,N>,2*N> prolongate_matrix()
{
  std::array<std::array<number,N>,2*N> RT{};
  for (auto k = 0u ; k < N ; k += p + 1)
    {
      bool avg = false ;
      std::size_t j = k ;
      for (auto i = 2*k ; i < 2*k+p+1 ; ++i)
	{
	  if (avg)
	    {
	      RT[i][j] = 0.5 ;
	      RT[i][j+1] = 0.5 ;
	      j += 1 ;
	      avg = false ;
	    }
	  else
	    {
	      RT[i][j] = 1. ;
	      avg = true ;
	    }
	}
      if (avg) avg = false ; else { avg = true ; j -= 1 ; }
      for (auto i = 2*k+p+1 ; i < 2*k+2*(p+1) ; ++i)
	{
	  if (avg)
	    {
	      RT[i][j] = 0.5 ;
	      RT[i][j+1] = 0.5 ;
	      j += 1 ;
	      avg = false ;
	    }
	  else
	    {
	      RT[i][j] = 1. ;
	      avg = true ;
	    }
	}
    }
  return RT;
}

template <std::size_t N,std::size_t p=1,std::size_t ...Is>
constexpr auto make_prolongation_impl(const std::index_sequence<Is...>)
{
  return std::make_tuple(prolongate_matrix<(N >> Is) * (p + 1) / 2,p>()...);
}

template <std::size_t N,std::size_t p=1,typename number=double>
constexpr auto make_prolongation()
{
  return make_prolongation_impl<N/2,p>(std::make_index_sequence<logg2(N/2) - 1>());
}

template <std::size_t N, typename Restriction, typename Prolongation, typename number=double>
class Multigrid
{

public:
  
  constexpr Multigrid(const std::array<std::array<number,N>,N> && A_,
		      const Restriction && rest_,
		      const Prolongation && prol_):
    A(std::forward<const std::array<std::array<number,N>,N> & >(A_)),
    rest(std::forward<const Restriction>(rest_)),
    prol(std::forward<const Prolongation>(prol_))
  {}

private:

  const std::array<std::array<number,N>,N> & A;
  const Restriction rest ;
  const Prolongation prol ;

public:
  
  template <std::size_t p=1, std::size_t n0=p+1, std::size_t s=1, std::size_t m=1,std::size_t t=0,
	    std::size_t nr, std::size_t nc>
  constexpr std::array<number,nr>
  vcycle0(const std::array<std::array<number,nc>,nr> && A,const std::array<number,nc> && res,
	  const number rlx=1.) const
  {
    if constexpr (nr == n0)
      {
	std::array<number,nr> q{},resq{};
	number norm = 1. ;
	while (true)
	  {
	    resq = res ;                                                       
	    sadd(resq,A*q,-1.) ;       
	    norm = std::sqrt(resq*resq) ;
	    if (norm < 1.E-12) break ;
	    sadd(q,invert(A)*resq) ;
	  }
	return q;
      }
    else
      {
	std::array<number,nr> x{},g{};
	const auto B = block_jacobi<p+1,t>(A) ;

	for (auto sit = 0;sit<s;++sit)
	  {
	    g = res ;                              //                   g
	    sadd(g,A*x,-1.);                       //                   g - A x_0
	    sadd(x,B*g,rlx);                       //    x_1 = x_0 + B (g - A x_0)
	  }

	g = res ;                        //                             g
	sadd(g,A*x,-1.);                 //                             g - A x_1

	const auto & R = std::get<logg2(N/nr)-1>(rest);
	const auto & RT = std::get<logg2(N/nr)-1>(prol);

	sadd(x,RT*
	     vcycle0
	     <p,n0,m*s,m,t>(std::forward<const std::array<std::array<number,nr/2>,nr/2> >(R*A*RT),
			    std::forward<const std::array<number,nr/2> >(R*g),rlx));
	                                 // q_1 = q_0 + RT A_0^{-1} (R (g - A x_1) - A_0 R q_0)
                                         //                    y_1 = x_1 + q_1

	for (auto sit = 0;sit<s;++sit)
	  {
	    g = res ;                              //                   g
	    sadd(g,A*x,-1.);                       //                   g - A y_1
	    sadd(x,B*g,rlx);                       //    y_2 = y_1 + B (g - A y_1)
	  }

	return x ;
      }
  }
  
  template <std::size_t p=1, std::size_t n0=p+1, std::size_t s=1, std::size_t m=1, std::size_t t=0,
	    bool prt=true, std::size_t nr, std::size_t nc>
  constexpr std::array<number,nr>
  vcycle(const std::array<std::array<number,nc>,nr> && A,const std::array<number,nc> && res,
	 const number rlx=1.) const
  {
    if constexpr (nr == n0)
      {
	if constexpr (prt) std::cout << nr/(p + 1) << std::flush;
	std::array<number,nr> q{},resq{};
	number norm = 1. ;
	while (true)
	  {
	    resq = res ;                                                       
	    sadd(resq,A*q,-1.) ;       
	    norm = std::sqrt(resq*resq) ;
	    if (norm < 1.E-12) break ;
	    sadd(q,vcycle0<p,n0,1,2>
		 (std::forward<typename std::remove_reference<decltype(A)>::type>(A),
		  std::forward<typename std::remove_reference<decltype(resq)>::type>(resq)));
	  }
	return q;
      }
    else
      {
	std::array<number,nr> x{},g{};
	const auto B = block_jacobi<p+1,t>(A) ;

	if constexpr (prt) std::cout << nr/(p + 1) << std::flush;
	if constexpr (prt and (s>1)) std::cout << "(" << s << ")" << std::flush ;
	for (auto sit = 0;sit<s;++sit)
	  {
	    g = res ;                              //                   g
	    sadd(g,A*x,-1.);                       //                   g - A x_0
	    sadd(x,B*g,rlx);                       //    x_1 = x_0 + B (g - A x_0)
	  }

	if constexpr (prt) std::cout << " \u2198 " << std::flush;
      
	g = res ;                        //                             g
	sadd(g,A*x,-1.);                 //                             g - A x_1
	const auto & R = std::get<logg2(N/nr)-1>(rest);
	const auto & RT = std::get<logg2(N/nr)-1>(prol);
	sadd(x,RT*
	     vcycle
	     <p,n0,m*s,m,t,prt>(std::forward
				<const std::array<std::array<number,nr/2>,nr/2> >(R*A*RT),
				std::forward
				<const std::array<number,nr/2> >(R*g),rlx));
                                         // q_1 = q_0 + RT A_0^{-1} (R (g - A x_1) - A_0 R q_0)
                                         //              y_1 = x_1 + q_1
      
	if constexpr (prt) std::cout << " \u2197 " <<  std::flush;
	for (auto sit = 0;sit<s;++sit)
	  {
	    g = res ;                              //                   g
	    sadd(g,A*x,-1.);                       //                   g - A y_1
	    sadd(x,B*g,rlx);                       //    y_2 = y_1 + B (g - A y_1)
	  }
	if constexpr (prt) std::cout << nr/(p + 1) << std::flush;
	if constexpr (prt and (s>1)) std::cout << "(" << s << ")" << std::flush ;

	return x ;
      }
  }

  template <std::size_t p=1,bool prt=true>
  constexpr auto solve(const std::array<number,N> && res, double rlx = 1.) const
  {
    return vcycle<p,p+1,1,2,0,prt>
      (std::forward<typename std::remove_reference<decltype(A)>::type>(A),
       std::forward<typename std::remove_reference<decltype(res)>::type>(res),
       rlx);
  }

};

template <std::size_t nr, std::size_t nc, std::size_t p=1,
	  bool rnd = true, bool prt=true, bool rndtest=false,typename number=double>
constexpr auto richardson(const std::array<std::array<number,nc>,nr> & A,
			  const std::array<number,nr> & rhs,
			  const number accuracy)
{
  auto itm = 0. ;
  auto itt = 0u ;
  while(true)
    {
      std::array<number,nr> x{},res{};

      if constexpr (rnd)
  	{
  	  std::uniform_real_distribution<double> unif(0.,1.);
  	  std::random_device re;
  	  for (auto & el : x) el = unif(re);
  	}

      sadd(res,rhs);
      sadd(res,A*x,-1.);                                  //                f - A x
      const Multigrid M(std::forward<typename std::remove_reference<decltype(A)>::type>(A),
      			make_restriction<2 * nc/(p + 1),p>(),make_prolongation<2 * nc/(p + 1),p>());

      number initial_norm = std::sqrt(res*res) ;
      number norm = 1. ;
      auto it = 0u;
      
      while (true)
      	{
      	  res = rhs ;                                     //                f
      	  sadd(res,A*x,-1.);                              //                f - A x
      	  norm = std::sqrt(res*res) ;                     //             \| f - A x \|
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
      	  sadd(x,M.template
      	       solve<p,prt>(std::forward<typename std::remove_reference<decltype(res)>::type>(res)));
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

template <std::size_t nel,std::size_t p=1,typename number=double>
constexpr auto mymain()
{
  return richardson<nel*(p+1),nel*(p+1),p,false>
    (assemble<nel,p>(static_cast<double>(p*(p+1)),1.),assemble_rhs<nel,p>(1.),1.E-8) ;
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

int main(int argc, char *argv[])
{
  const volatile auto x = make_main<256,3>();
  
  return 0;
}
