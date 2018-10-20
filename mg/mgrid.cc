#include <array>
#include <tuple>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>

template <std::size_t nr, typename number=double>
constexpr inline typename std::enable_if<std::is_arithmetic<number>::value,number>::type
operator *(const std::array<number,nr> & x, const std::array<number,nr> & y)
{
  number res = 0.;
  for (auto i=0u;i<x.size();++i)
    res += x[i]*y[i];
  return res ;
}

template <std::size_t nr, std::size_t nc, typename number=double>
constexpr
inline typename std::enable_if<std::is_arithmetic<number>::value,std::array<number,nr> >::type
operator *(const std::array<std::array<number,nc>,nr> & A,const std::array<number,nc> & x)
{
  std::array<number,nr> res{} ;
  for (auto i=0u;i<nr;++i)
    for (auto j=0u;j<nc;++j)
      res[i] += A[i][j]*x[j];
  return res ;
}

template
<std::size_t nrA, std::size_t ncA, std::size_t nrB, std::size_t ncB, typename number=double>
constexpr inline typename std::enable_if<std::is_arithmetic<number>::value,
				  std::array<std::array<number,ncB>,nrA> >::type
operator *(const std::array<std::array<number,ncA>,nrA> & A,
	   const std::array<std::array<number,ncB>,nrB> & B)
{
  std::array<std::array<number,ncB>,nrA> res{} ;
  for (auto k=0u;k<ncB;++k)
    for (auto i=0u;i<nrA;++i)
      for (auto j=0u;j<nrB;++j)      
	res[i][k] += A[i][j]*B[j][k];
  return res ;
}

template <std::size_t nr, std::size_t nc, typename number=double>
constexpr auto print(const std::array<std::array<number,nc>,nr> & A)
{
  for (auto i=0u;i<nr;++i)
    {
      for (auto j=0u;j<nc;++j)
  	std::cout << std::right << std::setw(11) << A[i][j];
      std::cout << std::endl ;
    }
  std::cout << std::endl ;
}

template <std::size_t n, typename number=double>
constexpr auto print(const volatile std::array<number,n> & v)
{
  for (const auto el : const_cast<std::array<number,n>& >(v))
    std::cout << el << " " ;
  std::cout << std::endl ;
}

constexpr inline auto sadd = [](auto & out, const auto & in, const double factor=1.)
{
  for (auto i=0u;i<in.size();++i) out[i]+=factor*in[i];
};

template <std::size_t n,typename number=double>
constexpr auto assemble(const number d,const number L=1.)
{
  std::array<std::array<number,2*n>,2*n> A{};
  const auto h = L/static_cast<number>(n);
  for (auto i = 0u;i<2*n;++i)
    {
      A[i][i] = d/h;
    }
  bool sw=true;
  for (auto i = 0u;i<2*n-2;++i)
    {
      A[i][i+2] = -1./(2.*h);
      if (sw) A[i+1][i+2] = -d/h+1./h;
      A[i+2][i] = -1./(2.*h);
      if (sw) A[i+2][i+1] = -d/h+1./h;
      if (sw) sw=false; else sw=true;
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

template <std::size_t n,typename number=double>
constexpr auto assemble_rhs()
{
  std::array<number,2*n> rhs{} ;
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
constexpr std::array<std::array<number,2*n>,n> restrict_matrix()
{
  std::array<std::array<number,2*n>,n> R{};
  R[0][0] = 1.;
  R[0][1] = 0.5;  
  R[1][1] = 0.5;  
  R[0][2] = 0.5;  
  R[1][2] = 0.5;
  
  auto i = 1u;
  for (auto j=3u;j<2*n-3;j+=4)
    {
      R[i][j] = 1.;
      R[i+1][j+1] = 1.;
      if (j<2*n-3)
	{
	  R[i+1][j+2] = 0.5;  
	  R[i+2][j+2] = 0.5;  
	  R[i+1][j+3] = 0.5;  
	  R[i+2][j+3] = 0.5;
	}
      i+=2;
    }
  R[n-1][2*n-1] = 1.;

  return R;
}

template <std::size_t N, std::size_t ...Is>
constexpr auto make_restriction_impl(const std::index_sequence<Is...>)
{
  return std::make_tuple(restrict_matrix<(N >> Is)>()...);
}

template <std::size_t N,typename number=double>
constexpr auto make_restriction(const std::array<std::array<number,N>,N>&)
{
  return make_restriction_impl<N/2>(std::make_index_sequence<logg2(N/2) - 1>());
}

template <std::size_t n,typename number=double>
constexpr std::array<std::array<number,n>,2*n> prolongate_matrix()
{
  std::array<std::array<number,n>,2*n> RT{};
  RT[0][0] = 1.;
  RT[1][0] = 0.5;  
  RT[1][1] = 0.5;  
  RT[2][0] = 0.5;  
  RT[2][1] = 0.5;
  
  auto i = 1u;
  for (auto j=3u;j<2*n-3;j+=4)
    {
      RT[j][i] = 1.;
      RT[j+1][i+1] = 1.;
      if (j<2*n-3)
	{
	  RT[j+2][i+1] = 0.5;  
	  RT[j+2][i+2] = 0.5;  
	  RT[j+3][i+1] = 0.5;  
	  RT[j+3][i+2] = 0.5;
	}
      i+=2;
    }
  RT[2*n-1][n-1] = 1.;

  return RT;
}

template <std::size_t N, std::size_t ...Is>
constexpr auto make_prolongation_impl(const std::index_sequence<Is...>)
{
  return std::make_tuple(prolongate_matrix<(N >> Is)>()...);
}

template <std::size_t N,typename number=double>
constexpr auto make_prolongation(const std::array<std::array<number,N>,N>&)
{
  return make_prolongation_impl<N/2>(std::make_index_sequence<logg2(N/2) - 1>());
}

template <std::size_t N, typename Restriction, typename Prolongation, typename number=double>
class Multigrid
{
public:
  
  constexpr Multigrid(const std::array<std::array<number,N>,N>&,
		      const Restriction && rest_,
		      const Prolongation && prol_):
    rest(std::forward<const Restriction>(rest_)),
    prol(std::forward<const Prolongation>(prol_))
  {}

private:
  
  const Restriction rest ;
  const Prolongation prol ;

public:
  
  template <std::size_t n0=2, std::size_t s=1, std::size_t m=1,std::size_t nr, std::size_t nc>
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
	    sadd(q,resq,1./A[0][0]) ;
	  }
	return q;
      }
    else
      {
	std::array<number,nr> x{},g{};

	for (auto sit = 0;sit<s;++sit)
	  {
	    g = res ;                    //                             g
	    sadd(g,A*x,-1.);             //                             g - A x_0
	    sadd(x,g,rlx/A[0][0]);       //              x_1 = x_0 + B (g - A x_0)
	  }
      
	g = res ;                        //                             g
	sadd(g,A*x,-1.);                 //                             g - A x_1

	const auto & R = std::get<logg2(N/nr)-1>(rest);
	const auto & RT = std::get<logg2(N/nr)-1>(prol);

	sadd(x,RT*
	     vcycle0
	     <n0,m*s,m>(std::forward<const std::array<std::array<number,nr/2>,nr/2> >(R*A*RT),
			std::forward<const std::array<number,nr/2> >(R*g),rlx));
	                                 // q_1 = q_0 + RT A_0^{-1} (R (g - A x_1) - A_0 R q_0)
                                         //                    y_1 = x_1 + q_1

	for (auto sit = 0;sit<s;++sit)
	  {
	    g = res ;                    //                             g
	    sadd(g,A*x,-1.);             //                             g - A y_1
	    sadd(x,g,rlx/A[0][0]);       //              y_2 = y_1 + B (g - A y_1)
	  }

	return x ;
      }
  }
  
  template <std::size_t n0=2, std::size_t s=1, std::size_t m=1, bool prt=true,
	    std::size_t nr, std::size_t nc>
  constexpr std::array<number,nr>
  vcycle(const std::array<std::array<number,nc>,nr> && A,const std::array<number,nc> && res,
	 const number rlx=1.) const
  {
    if constexpr (nr == n0)
      {
	if constexpr (prt) std::cout << nr/2 << std::flush;
	std::array<number,nr> q{},resq{};
	number norm = 1. ;
	while (true)
	  {
	    resq = res ;                                                       
	    sadd(resq,A*q,-1.) ;       
	    norm = std::sqrt(resq*resq) ;
	    if (norm < 1.E-12) break ;
	    sadd(q,vcycle0<2,1,2>
		 (std::forward<typename std::remove_reference<decltype(A)>::type>(A),
		  std::forward<typename std::remove_reference<decltype(resq)>::type>(resq)));
	  }
	return q;
      }
    else
      {
	std::array<number,nr> x{},g{};
	
	if constexpr (prt) std::cout << nr/2 << std::flush;
	if constexpr (prt and (s>1)) std::cout << "(" << s << ")" << std::flush ;
	for (auto sit = 0;sit<s;++sit)
	  {
	    g = res ;                    //                             g
	    sadd(g,A*x,-1.);             //                             g - A x_0
	    sadd(x,g,rlx/A[0][0]);       //              x_1 = x_0 + B (g - A x_0)
	  }
	if constexpr (prt) std::cout << " \u2198 " << std::flush;
      
	g = res ;                        //                             g
	sadd(g,A*x,-1.);                 //                             g - A x_1
	const auto & R = std::get<logg2(N/nr)-1>(rest);
	const auto & RT = std::get<logg2(N/nr)-1>(prol);

	sadd(x,RT*
	     vcycle
	     <n0,m*s,m,prt>(std::forward
			    <const std::array<std::array<number,nr/2>,nr/2> >(R*A*RT),
			    std::forward
			    <const std::array<number,nr/2> >(R*g),rlx));
                                         // q_1 = q_0 + RT A_0^{-1} (R (g - A x_1) - A_0 R q_0)
                                         //              y_1 = x_1 + q_1
      
	if constexpr (prt) std::cout << " \u2197 " <<  std::flush;
	for (auto sit = 0;sit<s;++sit)
	  {
	    g = res ;                    //                             g
	    sadd(g,A*x,-1.);             //                             g - A y_1
	    sadd(x,g,rlx/A[0][0]);       //              y_2 = y_1 + B (g - A y_1)
	  }
	if constexpr (prt) std::cout << nr/2 << std::flush;
	if constexpr (prt and (s>1)) std::cout << "(" << s << ")" << std::flush ;

	return x ;
      }
  }
};

template <std::size_t nr, std::size_t nc, bool rnd = true, bool prt=true, bool rndtest=false,
	  typename number=double>
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
      const Multigrid M(A,make_restriction(A),make_prolongation(A));
      number initial_norm = std::sqrt(res*res) ;
      number norm = 1. ;
      auto it = 0u;
      while (true)
	{
	  res = rhs ;                                     //                f
	  sadd(res,A*x,-1.);                              //                f - A x
	  norm = std::sqrt(res*res) ;                     //             \| f - A x \|
	  if constexpr (prt) std::cout.precision(3);
	  if constexpr (prt) std::cout << "\rN:" << nr/2 << "\tIteration: "<< it << "\t"
	  			       << initial_norm << " \u2192 " << norm << " "
	   			       << std::scientific << std::flush;
	  if (norm/initial_norm < accuracy) break;        // break
	  ++it ;
	  sadd(x,M.template vcycle<2,1,2,prt>
	       (std::forward<typename std::remove_reference<decltype(A)>::type>(A),
		std::forward<typename std::remove_reference<decltype(res)>::type>(res)));
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
